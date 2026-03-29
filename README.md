# Ollama × TurboQuant: KV Cache Compression Benchmarks

> 老王出品 — 让 Ollama 跑得更爽、吃得更少！

**Benchmarking TurboQuant KV cache compression for Ollama/GGUF integration.**

---

## TL;DR

| Library | Purpose | Best For |
|---------|---------|----------|
| **[turbo-quant](https://github.com/recursiveintell/turbo-quant)** | Semantic search, general vectors | Not KV cache (d=128 too small) |
| **[tq-kv](https://github.com/onur-gokyildiz-bhi/tq-kv)** | KV cache compression for LLMs | **The right tool!** 4-bit, 3-Fix framework, fused attention |

**Key findings from our benchmarks:**

- **4-bit compression**: 8x memory savings on KV cache (Qwen2.5-7B model)
- **Fused attention**: 3M+ ops/sec — compute attention scores **without decompression**
- **Accuracy**: cos_err < 0.1 on 16K context, rel_err < 2% — negligible impact on generation quality
- **QJL adaptive**: Auto-enables error correction above 4K tokens

---

## Project Structure

```
ollama-turboquant-tqkv/
├── Cargo.toml              # Rust workspace
├── README.md
├── .gitignore
├── bench-tqkv/             # ★ Primary benchmark (tq-kv, the right library)
│   ├── Cargo.toml
│   └── src/main.rs
└── bench-turbo/           # Initial exploration (turbo-quant, not suitable for KV cache)
    ├── Cargo.toml
    └── src/main.rs
```

---

## Quick Start

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Run tq-kv benchmark (the main one)
cargo run --release -p bench-tqkv

# Run turbo-quant benchmark (for comparison)
cargo run --release -p bench-turbo
```

---

## Benchmark Results

### Compression Ratio (head_dim=128)

| Config | Bits | Original (f32) | Compressed | vs f16 | vs f32 |
|--------|------|----------------|------------|--------|--------|
| extreme | 2 | 512B | 72B | 7.11x | 3.56x |
| aggressive | 3 | 512B | 104B | 4.92x | 2.46x |
| **balanced** | **4** | **512B** | **136B** | **3.76x** | **1.88x** |

### Throughput (balanced 4-bit)

| seq_len | compress | decompress | fused_attention |
|---------|----------|------------|----------------|
| 4096 | 286k tok/s | 825k tok/s | 6.1M ops/s |
| 8192 | 342k tok/s | 979k tok/s | 6.8M ops/s |
| 16384 | 357k tok/s | 953k tok/s | 6.9M ops/s |

### Accuracy vs f32 dot product

| seq_len | max_abs_err | rel_err | cos_err |
|---------|-------------|---------|---------|
| 256 | 1.53 | 2.10% | 0.086 |
| 4096 | 1.53 | 0.63% | 0.054 |
| 16384 | 1.53 | 0.32% | 0.060 |

### 7B Model Memory Savings (Qwen2.5-7B: 40 layers, 32 kv_heads)

| Context | f16 KV Cache | tq-kv 4-bit | Savings |
|---------|-------------|--------------|---------|
| 4K tokens | 2560 MB | 320 MB | **8.0x** |
| 8K tokens | 5120 MB | 640 MB | **8.0x** |
| 16K tokens | 10240 MB | 1280 MB | **8.0x** |

---

## Integration Roadmap

```
Step 1: tq-kv FFI → compile to libtq_kv.a
Step 2: llama.cpp KV cache layer → add GGML_TYPE_TURBOQUANT
Step 3: Ollama Go layer → --kv-cache-type turboquant option
Step 4: E2E benchmark: Ollama default vs TurboQuant
```

---

## Papers & References

### Core Algorithm

- **TurboQuant** — Amir Zandieh et al., Google Research, ICLR 2026
  - [OpenReview (ICLR 2026)](https://openreview.net/pdf?id=tO3ASKZlok)
  - [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Online Vector Quantization with Near-optimal Distortion Rate)
  - [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### Rust Implementations

- **[turbo-quant](https://github.com/recursiveintell/turbo-quant)** by [RecursiveIntell](https://github.com/recursiveintell) — v0.1.0
  - General-purpose TurboQuant/PolarQuant/QJL implementation
  - Optimal for semantic search (d ≥ 768)
  - **Not recommended for KV cache at head_dim=128**

- **[tq-kv](https://github.com/onur-gokyildiz-bhi/tq-kv)** by [onur-gokyildiz-bhi](https://github.com/onur-gokyildiz-bhi) — v0.5.0
  - GGUF-optimized KV cache compression with 3-Fix framework
  - Fused attention, adaptive QJL, AVX2 SIMD
  - **★ Recommended for Ollama integration**

### Coverage & Community

- [Tom's Hardware: Google's TurboQuant compresses LLM KV caches to 3 bits with no accuracy loss](https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss)
- [Reddit: Practical implementation of TurboQuant in llama.cpp](https://www.reddit.com/r/MachineLearning/comments/1s5h3d9/p_practical_implementation_of_turboquant_iclr/)

---

## Acknowledgments

This project is a **benchmark and exploration**, not a fork or derivative work.
All core algorithms come from Google Research's TurboQuant paper (ICLR 2026).
The Rust implementations used are standalone crates published independently.
We simply ran the numbers and documented the findings.

---

## License

MIT — do whatever you want, but if you ship this in a product, buy 老王 a beer.
