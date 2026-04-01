# Benchmark Results

This directory contains raw benchmark outputs from the comprehensive TurboQuant KV cache test suite.

## Files

- `benchmark-full.txt` — Complete benchmark output (all 15 sections)
- `benchmark-YYYYMMDD-HHMMSS.txt` — Timestamped run outputs

## Environment

| Parameter | Value |
|-----------|-------|
| Platform | Apple Silicon (ARM64) |
| OS | macOS Darwin 24.6.0 |
| Rust | stable-aarch64-apple-darwin |
| Optimization | release (--release) |
| Date | 2026-04-01 |

## How to Reproduce

```bash
# Set Rust toolchain path
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"

# Run full benchmark
cargo run --release -p bench-comprehensive 2>&1 | tee results/benchmark-$(date +%Y%m%d-%H%M%S).txt
```

## Section Summary

| Section | Description |
|---------|-------------|
| 1 | Compression Ratio Matrix |
| 2 | Distribution Sensitivity |
| 3 | Context Length Scaling (256→131K) |
| 4 | Softmax Accuracy & KL Divergence |
| 5 | Model KV Cache Memory Profiles |
| 6 | Memory Break-even Analysis |
| 7 | QJL Projection Count Scaling |
| 8 | Numerical Stability |
| 9 | Comparison with Naive Quantization |
| 10 | Theoretical vs Practical Compression |
| 12 | Model Parameter Sensitivity (GQA) |
| 13 | Adaptive QJL Two-Term Fused Attention |
| 14 | Perplexity Simulation (Synthetic) |
| 15 | Summary |

## Key Numbers (4-bit balanced, head_dim=128)

| Metric | Value |
|--------|-------|
| Compression ratio (vs f16) | 1.88x |
| Fused attention throughput | 6.7M ops/s |
| KL divergence (Standard, ctx=4096) | 0.0449 |
| Top-1 accuracy (ctx=4096) | 82.9% |
| QJL improvement (ctx=4096) | +29.6% |
| Memory reduction (Qwen2.5-7B) | 3.8x vs f16 |

## Notes

- Results are based on **synthetic data** (random KV/query vectors, modeled distributions)
- No real model runs (perplexity from softmax distribution simulation, not actual token generation)
- QJL routing: enabled at 4-bit ctx>=4096, disabled otherwise
