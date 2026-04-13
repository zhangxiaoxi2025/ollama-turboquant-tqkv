# P1 Dependencies: Real Model Validation

## Required for P1 Phase

### 1. Models (GGUF format, Q4_K_M quantized)

| Model | Size | Source |
|-------|------|--------|
| Qwen2.5-7B-Instruct | ~4.7GB | huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF |
| Llama-3.1-8B-Instruct | ~4.9GB | huggingface.co/MaziyarPanahi/Llama-3.1-8B-Instruct-GGUF |

**Recommended download:**
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# Or use Ollama
ollama pull qwen2.5:7b
```

### 2. Benchmark/Evaluation Tools

| Tool | Purpose | Install |
|------|---------|---------|
| lm-evaluation-harness | Standard perplexity benchmarks | `pip install lm-eval` |
| llama.cpp | Inference engine with profiling | Build from source |
| Ollama | End-to-end testing | `brew install ollama` |

### 3. Decode/Prefill Separation Test

**Option A: llama.cpp built-in profiler**
```bash
# Build llama.cpp with profiling
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Run with timing
./llama-cli -m model.gguf -p "test prompt" --timing-info
```

**Option B: Custom Rust benchmark**
- Measure time per token during decode phase
- Compare KV cache memory before/after compression
- Record attention score divergence at each layer

### 4. Long Context Tasks

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| RULER | Synthetic long-context retrieval | Accuracy vs context length |
| LongBench | Real-world long-context tasks | Multiple metrics |
| Needle-in-Haystack | Retrieval from long context | Recall@K |

**Minimal test:**
```bash
# Use RULER-like synthetic test
# Generate 4K-32K context with "needle" fact
# Query: "What is the [needle]?"
# Measure recall with/without KV compression
```

### 5. CI Scheme

**GitHub Actions workflow:**
```yaml
name: Benchmark
on: [push, pull_request]
jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo build --release -p bench-comprehensive
      - run: cargo run --release -p bench-comprehensive 2>&1 | head -200
```

**Note:** Real model validation requires ~10GB model files, not suitable for CI. CI should only verify:
- Build passes
- Synthetic benchmark runs
- No regressions in key metrics

## P1 Execution Plan

1. **Download model** (Qwen2.5-7B Q4_K_M GGUF)
2. **Run baseline perplexity** (no KV compression) - `lm-eval --model hf --model_args model.gguf --tasks wikitext`
3. **Apply KV compression** (tq-kv FFI in llama.cpp) - Requires backend integration
4. **Compare metrics:**
   - Perplexity delta
   - Decode tok/s
   - Memory footprint
   - Quality degradation at long context

## Blockers

- **P1.3 (Decode/Prefill test)**: Requires backend integration into llama.cpp
- **P1.4 (Long context tasks)**: Requires model + benchmark harness
- **Alternative**: Use synthetic Section 14 as proxy until backend is ready
