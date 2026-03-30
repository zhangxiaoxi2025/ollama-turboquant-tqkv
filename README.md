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

# Run comprehensive benchmark (15 test sections: distribution sensitivity, softmax accuracy, QJL scaling, adaptive two-term attention, synthetic perplexity, etc.)
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
├── bench-comprehensive/    # Extended benchmark (15 test sections)
│   ├── Cargo.toml
│   └── src/main.rs        # compression, distribution, softmax, QJL, perplexity...
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

### Attention Score Accuracy (vs FP32 dot product, balanced 4-bit, adaptive QJL)

| Sequence Length | QJL Mode | KL Divergence | Top-1 Accuracy | Quality |
|----------------|----------|---------------|----------------|---------|
| 256 tokens | OFF | 0.039 | 86.7% | Acceptable |
| 1,024 tokens | OFF | 0.045 | 84.4% | Acceptable |
| 4,096 tokens | ON (+29.6%) | 0.045 | 82.9% | Acceptable |
| 16,384 tokens | ON | 0.053 | — | Acceptable |

**Routing logic**: QJL activates at `bits == 4 AND context_length >= 4096`. Short context skips QJL — the variance cost of QJL sketches outweighs error correction benefit. At 4-bit + 4K+ tokens, QJL delivers +29.6% MSE improvement on Standard distributions (but may harm DeepLayer/Sparse by -7% to -11%, which the adaptive routing accepts for aggregate benefit).

### Model Memory Analysis (Qwen2.5-7B: 40 layers, 32 KV heads, head_dim=128)

| Context Length | FP16 KV Cache | tq-kv 4-bit | Reduction Factor | Memory Saved |
|----------------|---------------|--------------|-----------------|-------------|
| 1,024 tokens | 640 MB | 170 MB | 3.8x | 470 MB |
| 2,048 tokens | 1,280 MB | 340 MB | 3.8x | 940 MB |
| 4,096 tokens | 2,560 MB | 680 MB | 3.8x | 1,880 MB |
| 8,192 tokens | 5,120 MB | 1,360 MB | 3.8x | 3,760 MB |
| 16,384 tokens | 10,240 MB | 2,720 MB | 3.8x | 7,520 MB |

### Adaptive QJL Two-Term Fused Attention

The two-term fused attention formula combines MSE term (codebook centroid projection) with an optional QJL term (unbiased residual sketch):

```
score_i = <rotated_q, centroids_i> + alpha * <rotated_q, H @ D @ signs_i>
```

| Condition | Mode | QJL Effect |
|-----------|------|------------|
| bits < 4 (2-bit, 3-bit) | MSE-only | QJL disabled — negative effect on low-bit distributions |
| bits == 4, ctx < 4096 | MSE-only | QJL disabled — variance cost exceeds benefit |
| bits == 4, ctx >= 4096 | Two-term | QJL +29.6% MSE improvement on Standard |

**Distribution-aware note**: QJL improves Standard/FlashLike but harms DeepLayer/Sparse even at 4-bit ctx>=4096. The routing makes a conservative aggregate trade-off (Standard is the most common distribution in practice).

### Key Findings

1. **head_dim=128**: optimal bits=4, break-even at ~175 tokens/layer (4-bit)
2. **GQA models benefit most**: 8:1 KV head ratio → KV cache is 3.8x smaller
3. **KL divergence by bit-width**: 2-bit ≈ 0.48–0.72, 3-bit ≈ 0.15–0.22, 4-bit ≈ 0.04–0.06
4. **Top-1 accuracy by bit-width**: 2-bit ≈ 48–66%, 3-bit ≈ 68–74%, 4-bit ≈ 83–87%
5. **QJL effect**: MSE score improvement +29.6% on Standard at 4-bit ctx=4096; KL domain improvement ~1% (marginal)
6. **Adaptive routing**: QJL enables at 4-bit ctx>=4096, disabled otherwise — prevents negative effect on 2/3-bit and short-context
7. **DeepLayer distribution**: higher error than Standard, still acceptable at 4-bit
8. **Sparse distribution**: most challenging (KL ≈ 0.07–0.15 at 4-bit), accuracy degrades at long context
9. **Numerical stability**: KL divergence handles zero/NaN gracefully; cosine similarity clamped to [-1, 1]
10. **vs Naive quantization**: tq-kv 5–8x better cos_err at same bits (84–98% rel_err improvement)

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
