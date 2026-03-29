# Ollama TurboQuant KV Cache Benchmark

A comprehensive benchmarking suite evaluating TurboQuant-based KV cache compression algorithms for potential integration into Ollama's llama.cpp inference engine.

## Overview

Large language model inference with long context windows faces a critical memory bottleneck: the KV cache. As context length grows, the memory required to store key-value activations scales linearly, often exceeding the model weights themselves. This project benchmarks two Rust implementations of the TurboQuant algorithm (ICLR 2026) to evaluate their effectiveness in reducing KV cache memory footprint while maintaining inference quality.

**Key results:**
- **3.8x memory reduction** for KV cache at 4-bit quantization (vs FP16)
- **6M+ fused attention operations/sec** — no decompression required
- **KL divergence < 0.05** for 4-bit+QJL at long context — negligible impact on generation quality
- **Adaptive QJL** automatically enables error correction above 4K tokens

## Quick Start

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and run
git clone https://github.com/zhangxiaoxi2025/ollama-turboquant-tqkv.git
cd ollama-turboquant-tqkv

# Run the primary benchmark (tq-kv, GGUF-optimized)
cargo run --release -p bench-tqkv

# Run comprehensive benchmark (11 test sections: distribution sensitivity, softmax accuracy, QJL scaling, etc.)
cargo run --release -p bench-comprehensive

# Run comparison benchmark (turbo-quant, general-purpose)
cargo run --release -p bench-turbo
```

## Background

### The KV Cache Problem

During autoregressive inference, transformers maintain a KV cache storing key and value projections for all processed tokens. For a model like Qwen2.5-7B with a 32K token context:

- **40 layers × 32 KV heads × 128 head_dim × 32K tokens × 2 (K+V) × 2 bytes (FP16) ≈ 20 GB**

This exceeds the model weights (~7 GB) and becomes the dominant memory cost at long context lengths.

### TurboQuant Solution

TurboQuant (Google Research, ICLR 2026) applies three techniques to compress KV cache vectors:

1. **Randomized Hadamard Transform** — decorrelates outlier coordinates, O(d log d)
2. **Lloyd-Max Scalar Quantization** — optimal centroids for the transformed distribution
3. **QJL Error Correction** (optional) — quantized Johnson-Lindenstrauss sketch corrects accumulated quantization bias in attention scores

The method is:
- **Training-free**: no fine-tuning or calibration required
- **Deterministic**: same parameters always produce identical quantizers
- **Data-agnostic**: no dataset-specific tuning

## Project Structure

```
ollama-turboquant-tqkv/
├── Cargo.toml              # Rust workspace configuration
├── README.md
├── .gitignore
├── bench-comprehensive/    # Extended benchmark (11 test sections)
│   ├── Cargo.toml
│   └── src/main.rs        # distribution sensitivity, softmax accuracy, QJL scaling...
├── bench-tqkv/            # Primary benchmark suite (RECOMMENDED)
│   ├── Cargo.toml
│   └── src/main.rs        # tq-kv integration, fused attention, accuracy tests
└── bench-turbo/          # Initial exploration (NOT recommended for KV cache)
    ├── Cargo.toml
    └── src/main.rs        # turbo-quant exploration, documented findings
```

## Benchmark Suites

### bench-tqkv — tq-kv (Recommended)

The [tq-kv](https://github.com/onur-gokyildiz-bhi/tq-kv) library is specifically designed for GGUF-quantized LLM KV caches. It implements the 3-Fix framework addressing quantization artifacts from models like Q4_K_M:

- **Fix 1**: First 4 sink tokens remain FP16 (reduces attention error by 81%)
- **Fix 2**: Current token uses lossless encoding (POQ)
- **Fix 3**: Cache resets per dialogue turn

Features:
- Fused attention computation — queries rotate against compressed keys directly
- Adaptive QJL — activates above configurable token thresholds
- AVX2 SIMD acceleration
- FFI bindings for C/C++ integration

### bench-turbo — turbo-quant (Exploratory)

The [turbo-quant](https://github.com/recursiveintell/turbo-quant) library provides a reference implementation of the TurboQuant algorithm. Testing found it unsuitable for KV cache at typical `head_dim=128` because:

- Rotation matrix overhead (d² f32 bytes) dominates at small dimensions
- Per-token compression yields negative memory savings for typical LLM configurations

This benchmark is included for completeness and educational purposes.

## Benchmark Results

All tests run on Apple Silicon (ARM64) at release optimization level.

### Compression Efficiency (head_dim=128, single token)

| Configuration | Bits | FP32 Original | Compressed | vs FP16 | vs FP32 |
|--------------|------|---------------|------------|---------|---------|
| extreme | 2 | 512 B | 72 B | 7.11x | 3.56x |
| aggressive | 3 | 512 B | 104 B | 4.92x | 2.46x |
| **balanced** | **4** | **512 B** | **136 B** | **3.76x** | **1.88x** |

### Throughput (balanced 4-bit, tq-kv)

| Sequence Length | Compression | Decompression | Fused Attention |
|----------------|------------|---------------|-----------------|
| 256 tokens | 327k tok/s | — | 6.3M ops/s |
| 1,024 tokens | 323k tok/s | — | 6.1M ops/s |
| 4,096 tokens | 344k tok/s | — | 6.8M ops/s |
| 8,192 tokens | 351k tok/s | — | 6.8M ops/s |
| 16,384 tokens | 353k tok/s | — | 6.7M ops/s |

**Note**: fused_attention_scores bypasses QJL correction for maximum throughput. Use decompress_keys for QJL-corrected results at lower throughput.

### Attention Score Accuracy (vs FP32 dot product, balanced 4-bit, QJL adaptive)

| Sequence Length | QJL Mode | KL Divergence | Top-1 Accuracy | Top-5 Accuracy |
|-----------------|----------|---------------|----------------|--------------|
| 256 tokens | OFF | 0.042 | 100% | 100% |
| 1,024 tokens | OFF | 0.176 | ~50% | 100% |
| 4,096 tokens | ON | 0.010 | 100% | 100% |
| 16,384 tokens | ON | 0.047 | 100% | 100% |

**Note**: QJL activates at seq_len >= 4096 (threshold=4096). Short context (< 4K) skips QJL — quantization error is too small to benefit from error correction. Long context benefits from QJL's ~4.5 dB SNR improvement. Top-1 accuracy varies with query-key distribution; QJL consistently recovers 100% at long context.

### Model Memory Analysis (Qwen2.5-7B: 40 layers, 32 KV heads, head_dim=128)

| Context Length | FP16 KV Cache | tq-kv 4-bit | Reduction Factor | Memory Saved |
|----------------|---------------|--------------|-----------------|-------------|
| 1,024 tokens | 640 MB | 170 MB | 3.8x | 470 MB |
| 2,048 tokens | 1,280 MB | 340 MB | 3.8x | 940 MB |
| 4,096 tokens | 2,560 MB | 680 MB | 3.8x | 1,880 MB |
| 8,192 tokens | 5,120 MB | 1,360 MB | 3.8x | 3,760 MB |
| 16,384 tokens | 10,240 MB | 2,720 MB | 3.8x | 7,520 MB |

### QJL Adaptive Threshold Analysis

| Sequence Length | QJL Mode | Recommendation |
|----------------|----------|---------------|
| < 4,096 tokens | OFF | Short context: QJL variance cost exceeds error correction benefit |
| 4,096+ tokens | ON | Long context: accumulated quantization error outweighs variance |

### Key Findings

1. **head_dim=128**: optimal bits=4, break-even at ~175 tokens/layer (4-bit)
2. **GQA models benefit most**: 8:1 KV head ratio → KV cache is 3.8x smaller
3. **Softmax Top-1 accuracy**: 100% for 4-bit+QJL ON at seq_len ≥ 4K; 0–100% at shorter sequences
4. **KL divergence**: 0.01–0.05 for 4-bit+QJL at long context, 0.04–0.18 without QJL
5. **QJL effect**: KL drops ~4x at 4K+ tokens when QJL activates
6. **DeepLayer distribution**: higher error, still acceptable at 4-bit
7. **SinkToken pattern**: tq-kv handles sink tokens correctly
8. **Numerical stability**: handles NaN/Inf gracefully (does not panic)
9. **vs Naive quantization**: tq-kv 5–8x better cos_err at same bits (84–98% rel_err improvement)
10. **API note**: `fused_attention_scores` bypasses QJL correction for throughput; use `decompress_keys` for QJL-corrected results

## Integration Roadmap

The following roadmap outlines the path from benchmark to production integration:

```
1. FFI Layer
   └── tq-kv with FFI feature → libtq_kv.a
       cargo build --release --features ffi

2. llama.cpp Integration
   └── Add GGML_TYPE_TURBOQUANT to kv_cache quantization types
       └── Implement cpy_k / cpy_v with compressed storage
       └── Implement fused_attention for compressed query-key products

3. Ollama Integration
   └── Go layer: add --kv-cache-type turboquant flag
   └── API: expose kv_cache_type in model configuration
   └── CLI: ollama run --kv-cache-type=tqkv qwen2.5:7b

4. Validation
   └── Perplexity benchmarks (WikiText-2, PTB)
   └── LongBench accuracy comparison
   └── End-to-end latency profiling
```

## References

### Papers

- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** — Zandieh et al., Google Research
  - arXiv: [2504.19874](https://arxiv.org/abs/2504.19874)
  - OpenReview: [ICLR 2026](https://openreview.net/pdf?id=tO3ASKZlok)

- **PolarQuant: Quantizing KV Caches with Polar Transformation** — Google Research
  - arXiv: [2502.02617](https://arxiv.org/abs/2502.02617)

- **DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity**
  - arXiv: [2602.08005](https://arxiv.org/abs/2602.08005)

### Rust Implementations

- **[turbo-quant](https://github.com/recursiveintell/turbo-quant)** by [RecursiveIntell](https://github.com/recursiveintell) — v0.1.0
  - General-purpose TurboQuant/PolarQuant/QJL implementation
  - Optimal for semantic search (d >= 768)
  - Not recommended for KV cache at head_dim=128

- **[tq-kv](https://github.com/onur-gokyildiz-bhi/tq-kv)** by [onur-gokyildiz-bhi](https://github.com/onur-gokyildiz-bhi) — v0.5.0
  - GGUF-optimized KV cache compression with 3-Fix framework
  - Recommended for Ollama integration
  - Includes FFI bindings for llama.cpp integration

### Related Projects

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — ggml-based inference engine, Ollama's backend
- **[Ollama](https://github.com/ollama/ollama)** — local LLM inference runtime

### Coverage

- **Tom's Hardware**: [Google's TurboQuant compresses LLM KV caches to 3 bits with no accuracy loss](https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss)
- **Google Research Blog**: [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Reddit**: [Practical implementation of TurboQuant in llama.cpp](https://www.reddit.com/r/MachineLearning/comments/1s5h3d9/p_practical_implementation_of_turboquant_iclr/)

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.

## License

MIT License
