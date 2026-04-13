# Benchmark Environment

## Hardware
- Platform: Apple Silicon (ARM64)
- OS: macOS Darwin 24.6.0

## Software
- Rust: stable-aarch64-apple-darwin
- Optimization: release (--release)
- Date: 2026-04-01

## Dependencies
- tq-kv = "0.5"
- rand = "0.8"

## How to Reproduce

```bash
# Set Rust toolchain path
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"

# Run full benchmark (WARNING: Section 14 takes 10-30 minutes)
cargo run --release -p bench-comprehensive 2>&1 | tee results/benchmark-$(date +%Y%m%d-%H%M%S).txt

# Quick run (skip Section 14)
# Edit src/main.rs to comment out test_perplexity_simulation() call
```

## Section Summary

| Section | Description |
|---------|-------------|
| 1 | Compression Ratio Matrix |
| 2 | Distribution Sensitivity |
| 3 | Context Length Scaling (256 → 131K) |
| 4 | Softmax Accuracy & KL Divergence |
| 5 | Model KV Cache Memory Profiles |
| 6 | Memory Break-even Analysis |
| 7 | QJL Projection Count Scaling |
| 8 | Numerical Stability |
| 9 | Comparison with Naive Quantization |
| 10 | Theoretical vs Practical Compression |
| 12 | Model Parameter Sensitivity (GQA) |
| 13 | Adaptive QJL Two-Term Fused Attention |
| 14 | Perplexity Simulation (Synthetic) - O(n^2), slow |
| 15 | Summary |

## Key Numbers (4-bit balanced, head_dim=128)

| Metric | Value |
|--------|-------|
| Compression ratio (vs f16) | 1.88x |
| Fused attention throughput | ~6.7M ops/s |
| KL divergence (Standard, ctx=4096) | 0.0449 |
| Top-1 accuracy (ctx=4096) | 82.9% |
| QJL MSE improvement (ctx=4096, Standard) | +29.6% |
| Memory reduction (Qwen2.5-7B, 4-bit) | 3.8x vs f16 |

## Notes

- Results are based on **synthetic data** (random KV/query vectors, modeled distributions)
- No real model perplexity validation included
- Section 14 is computationally expensive (O(n^2) for full perplexity simulation)
