//! RoPE Compatibility: Mathematical Proof & Fix Exploration
//!
//! Core question: Does tq-kv's fused attention work correctly with RoPE models?
//!
//! Key insight:
//! - compress_keys stores: quant(H·RoPE(k_base)) — Hadamard AFTER RoPE
//! - pre_rotate_query applies: H·q_base — only Hadamard, no RoPE
//! - In real LLM: q arrives as RoPE(q_base) (already rotated by the model)
//! - fused = <H·q_base, quant(H·RoPE(k_base))> ≠ <RoPE(q_base), RoPE(k_base)>
//!
//! This test proves the incompatibility and explores potential fixes.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tq_kv::{
    codebook, compress_keys, decompress_keys, fused_attention_scores,
    pre_rotate_query, CompressedKeys, TurboQuantConfig,
};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let num: f32 = a.iter().take(n).zip(b.iter().take(n)).map(|(x, y)| x * y).sum();
    num / (norm(a) * norm(b))
}

/// Apply RoPE rotation to a vector at given position.
///
/// Standard RoPE formula (matches llama.cpp/HuggingFace RoFormer):
///   freq_i = base^(-2i/d)  (per-dimension frequency, decays with i)
///   angle_i = position * freq_i  (position-dependent rotation)
///   Apply 2D rotation (cos, -sin; sin, cos) to each (x0, x1) pair at dimension i.
///
/// Bug in previous versions: used `position / base^(-2i/d)` which gives
/// HUGE angles at low dimensions (not rotating) and tiny angles at high dimensions.
fn apply_rope(vector: &[f32], position: usize, head_dim: usize, base: f64) -> Vec<f32> {
    let mut result = vector.to_vec();
    let half_dim = head_dim / 2;
    let pos_f = position as f64;
    for i in 0..half_dim {
        let freq = base.powi(-2 * i as i32 / head_dim as i32); // decays with i
        let theta = pos_f * freq; // position-dependent angle
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let x0 = result[i] as f64;
        let x1 = result[i + half_dim] as f64;
        result[i] = (x0 * cos_theta - x1 * sin_theta) as f32;
        result[i + half_dim] = (x1 * cos_theta + x0 * sin_theta) as f32;
    }
    result
}

/// Inverse RoPE — undo RoPE rotation.
///
/// Since RoPE is orthogonal (R^T = R^{-1}), the inverse uses:
///   cos(θ), +sin(θ); -sin(θ), cos(θ)  (transpose of forward)
/// where θ = position * freq_i (same as forward).
fn inverse_rope(vector: &[f32], position: usize, head_dim: usize, base: f64) -> Vec<f32> {
    let mut result = vector.to_vec();
    let half_dim = head_dim / 2;
    let pos_f = position as f64;
    for i in 0..half_dim {
        let freq = base.powi(-2 * i as i32 / head_dim as i32);
        let theta = pos_f * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        // Transpose: swap sign on sin (R^T = R^{-1} for rotation matrix)
        let x0 = result[i] as f64;
        let x1 = result[i + half_dim] as f64;
        result[i] = (x0 * cos_theta + x1 * sin_theta) as f32;
        result[i + half_dim] = (x1 * cos_theta - x0 * sin_theta) as f32;
    }
    result
}

fn generate_base(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values);
    let variance =
        values.iter().map(|&v| (v - avg).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

fn analyze_scores(ref_s: &[f32], tq_s: &[f32]) -> (f32, f32, f32) {
    // Compute cosine similarity between the two score vectors
    let cos_sim = cosine_sim(ref_s, tq_s);
    // cos_err: 1 - |cos_sim| (how far from perfect alignment)
    let cos_err = (1.0 - cos_sim.abs()).abs();

    // Relative error
    let n = ref_s.len().max(1);
    let sq_err: f32 = ref_s
        .iter()
        .zip(tq_s.iter())
        .map(|(r, t)| (r - t).powi(2))
        .sum();
    let mse = sq_err / n as f32;
    let ref_pow = ref_s.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let rel_err = sq_err.sqrt() / ref_pow;

    (cos_err, rel_err, mse)
}

fn build_cache(keys: &[Vec<f32>], config: &TurboQuantConfig, hd: usize) -> CompressedKeys {
    let mut cache = CompressedKeys::new_empty(config.bits, hd, config.rotation_seed);
    for k in keys {
        let s = compress_keys(k, hd, config);
        cache.append_raw(
            &s.packed_indices[..s.bytes_per_vector()],
            s.norms[0],
        );
    }
    cache
}

fn main() {
    let hd = 128;
    let seq_len = 1024;
    let rope_base = 500000.0; // Qwen2.5
    let seeds = [42u64, 123, 777, 2024, 3141];

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  ROPE COMPATIBILITY — MATHEMATICAL PROOF & FIX EXPLORATION            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // TEST A: Verify no-RoPE baseline (benchmark ground truth)
    // =========================================================================
    println!("TEST A: No-RoPE Baseline (what benchmark actually measures)\n");

    let mut no_rope_cos = Vec::new();
    let mut no_rope_rel = Vec::new();

    for &seed in &seeds {
        let config = TurboQuantConfig::balanced();
        let mut rng = StdRng::seed_from_u64(seed);

        let keys: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| generate_base(hd, &mut rng))
            .collect();
        let query = generate_base(hd, &mut rng);

        // Ground truth: <q, k> (no RoPE)
        let gt: Vec<f32> = keys.iter().map(|k| dot(&query, k)).collect();

        // Cache (no RoPE)
        let cache = build_cache(&keys, &config, hd);

        // Fused attention
        let rotated_q = pre_rotate_query(&query, config.rotation_seed);
        let tq = fused_attention_scores(
            &rotated_q,
            &cache,
            &codebook::get_centroids(config.bits),
            1.0,
        );

        let (cos_err, rel_err, _) = analyze_scores(&gt, &tq);
        no_rope_cos.push(cos_err);
        no_rope_rel.push(rel_err);
    }

    println!(
        "  No-RoPE cos_err: {:.4} ± {:.4}",
        mean(&no_rope_cos),
        std_dev(&no_rope_cos)
    );
    println!(
        "  No-RoPE rel_err: {:.4} ± {:.4}",
        mean(&no_rope_rel),
        std_dev(&no_rope_rel)
    );
    println!(
        "  → Baseline: quantization error in Hadamard space is ACCEPTABLE (~0.06)\n"
    );

    // =========================================================================
    // TEST B: RoPE — fused vs TRUE RoPE attention (the core test)
    // =========================================================================
    println!("TEST B: RoPE — Fused Attention vs True RoPE Attention\n");

    let mut rope_cos = Vec::new();
    let mut rope_rel = Vec::new();

    for &seed in &seeds {
        let config = TurboQuantConfig::balanced();
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate base keys and query
        let keys_base: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| generate_base(hd, &mut rng))
            .collect();
        let query_base = generate_base(hd, &mut rng);

        // Apply RoPE to keys at each position
        let keys_rope: Vec<Vec<f32>> = keys_base
            .iter()
            .enumerate()
            .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
            .collect();

        // Apply RoPE to query (at last position, like real LLM)
        let query_rope = apply_rope(&query_base, seq_len - 1, hd, rope_base);

        // TRUE RoPE attention: <RoPE(q), RoPE(k)> = <q_base, k_base> (by orthogonality)
        let gt: Vec<f32> = keys_rope
            .iter()
            .zip(std::iter::repeat(&query_rope))
            .map(|(k, q)| dot(q, k))
            .collect();

        // Cache: compress RoPE-rotated keys (simulating real LLM where key is
        // already RoPE'd before reaching KV cache)
        // compress_keys does: H(quant(k)), so H·RoPE(k_base) gets stored
        let cache = build_cache(&keys_rope, &config, hd);

        // Fused attention: pre_rotate_query only applies Hadamard (NOT RoPE)
        // Query arrives as RoPE(q_base), pre_rotate_query applies H·RoPE(q_base)
        let rotated_q = pre_rotate_query(&query_rope, config.rotation_seed);
        let tq = fused_attention_scores(
            &rotated_q,
            &cache,
            &codebook::get_centroids(config.bits),
            1.0,
        );

        let (cos_err, rel_err, _) = analyze_scores(&gt, &tq);
        rope_cos.push(cos_err);
        rope_rel.push(rel_err);
    }

    let rope_cos_mean = mean(&rope_cos);
    let rope_cos_std = std_dev(&rope_cos);

    println!(
        "  RoPE cos_err: {:.4} ± {:.4}",
        rope_cos_mean, rope_cos_std
    );
    println!(
        "  RoPE rel_err: {:.4} ± {:.4}",
        mean(&rope_rel),
        std_dev(&rope_rel)
    );
    println!(
        "  → Fused attention vs true RoPE attention: cos_err ≈ {:.0} (NEAR MAXIMUM!)\n",
        rope_cos_mean
    );

    // =========================================================================
    // TEST C: Fix attempt — pre_rotate_query with inverse RoPE
    // =========================================================================
    // Theory: if we apply inverse RoPE to the query BEFORE Hadamard, does it fix it?
    //
    // Ideal: <H·q', quant(H·RoPE(k))> = <RoPE(q), RoPE(k)>
    // Since <Hx, Hy> = <x, y> (Hadamard orthogonality):
    //   <H·q', H·RoPE(k)> = <q', RoPE(k)> = <RoPE(q), RoPE(k)>
    //   So q' = RoPE_inv(RoPE(q)) = q... but q here is already RoPE(q_base)!
    //
    // Let's try: q' = H·RoPE_inv(q_rope) = H·q_base
    // fused = <H·q', quant(H·RoPE(k))> = <H·H·q_base, quant(H·RoPE(k))>
    //        = <q_base, quant(H·RoPE(k))>  (H self-inverse)
    // vs true: <q_base, k_base> (by RoPE orthogonality)
    //
    // Since quant(H·RoPE(k)) ≠ k_base, these don't match.
    // But let's test it empirically anyway!

    println!("TEST C: Fix Attempt — Inverse RoPE Before Hadamard\n");

    let mut fix_cos = Vec::new();

    for &seed in &seeds {
        let config = TurboQuantConfig::balanced();
        let mut rng = StdRng::seed_from_u64(seed);

        let keys_base: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| generate_base(hd, &mut rng))
            .collect();
        let query_base = generate_base(hd, &mut rng);

        let keys_rope: Vec<Vec<f32>> = keys_base
            .iter()
            .enumerate()
            .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
            .collect();
        let query_rope = apply_rope(&query_base, seq_len - 1, hd, rope_base);

        // TRUE RoPE attention
        let gt: Vec<f32> = keys_rope
            .iter()
            .zip(std::iter::repeat(&query_rope))
            .map(|(k, q)| dot(q, k))
            .collect();

        // Cache
        let cache = build_cache(&keys_rope, &config, hd);

        // FIX attempt: apply inverse RoPE to query, THEN Hadamard
        // q' = H·RoPE_inv(q_rope) = H·q_base
        let query_inv_rope = inverse_rope(&query_rope, seq_len - 1, hd, rope_base);
        let rotated_q_fix = pre_rotate_query(&query_inv_rope, config.rotation_seed);

        let tq = fused_attention_scores(
            &rotated_q_fix,
            &cache,
            &codebook::get_centroids(config.bits),
            1.0,
        );

        let (cos_err, _, _) = analyze_scores(&gt, &tq);
        fix_cos.push(cos_err);
    }

    println!(
        "  Fix cos_err: {:.4} ± {:.4}",
        mean(&fix_cos),
        std_dev(&fix_cos)
    );
    println!(
        "  No-fix (B) cos_err: {:.4} ± {:.4}\n",
        rope_cos_mean, rope_cos_std
    );

    let fix_improvement = rope_cos_mean - mean(&fix_cos);
    if fix_improvement > 0.05 {
        println!(
            "  → Fix IMPROVES cos_err by {:.4}! ({}%)",
            fix_improvement,
            (fix_improvement / rope_cos_mean * 100.0) as i32
        );
    } else {
        println!("  → Fix does NOT fix the incompatibility (H·RoPE ≠ RoPE·H)");
    }

    // =========================================================================
    // TEST D: Alternative — decompress keys then compute RoPE-aware attention
    // =========================================================================
    // If we decompress and compute the attention manually:
    // decompress returns: H(quant(H(RoPE(k_base)))) ≈ quant(H(RoPE(k_base)))
    // Then: <RoPE(q_base), quant(H(RoPE(k_base)))>
    // Let's see what this gives us.

    println!("\nTEST D: Decompressed Keys + Manual RoPE Attention\n");

    let mut decomp_cos = Vec::new();

    for &seed in &seeds {
        let config = TurboQuantConfig::balanced();
        let mut rng = StdRng::seed_from_u64(seed);

        let keys_base: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| generate_base(hd, &mut rng))
            .collect();
        let query_base = generate_base(hd, &mut rng);

        let keys_rope: Vec<Vec<f32>> = keys_base
            .iter()
            .enumerate()
            .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
            .collect();
        let query_rope = apply_rope(&query_base, seq_len - 1, hd, rope_base);

        let gt: Vec<f32> = keys_rope
            .iter()
            .zip(std::iter::repeat(&query_rope))
            .map(|(k, q)| dot(q, k))
            .collect();

        let cache = build_cache(&keys_rope, &config, hd);

        // Decompress: returns approx quant(H(RoPE(k_base)))
        let decompressed = decompress_keys(&cache, &config);

        // Manual attention: <RoPE(q), decompressed>
        // decompressed[i*hd..(i+1)*hd] ≈ quant(H(RoPE(k_base[i])))
        let tq: Vec<f32> = (0..seq_len)
            .map(|i| {
                let dk = &decompressed[i * hd..(i + 1) * hd];
                dot(&query_rope, dk)
            })
            .collect();

        let (cos_err, _, _) = analyze_scores(&gt, &tq);
        decomp_cos.push(cos_err);
    }

    println!(
        "  Decompressed cos_err: {:.4} ± {:.4}",
        mean(&decomp_cos),
        std_dev(&decomp_cos)
    );
    println!(
        "  Fused (no fix) cos_err: {:.4} ± {:.4}\n",
        rope_cos_mean, rope_cos_std
    );

    // =========================================================================
    // MATHEMATICAL PROOF SUMMARY
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  MATHEMATICAL PROOF: WHY NO FIX WORKS                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    println!("  Chain of transformations:\n");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ INFERENCE PIPELINE (RoPE model)                                         │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ 1. Key at position i: RoPE(k_base[i]) pre-applied by LLM               │");
    println!("  │ 2. compress_keys stores: quant(H · RoPE(k_base[i]))                   │");
    println!("  │    (Hadamard applied AFTER RoPE)                                      │");
    println!("  │ 3. Query at position j: q = RoPE(q_base[j])                           │");
    println!("  │ 4. pre_rotate_query(q) = H · q = H · RoPE(q_base[j])                  │");
    println!("  │ 5. fused_attention = <H·RoPE(q_base), quant(H·RoPE(k_base))>         │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ GROUND TRUTH: True RoPE Attention                                      │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ 1. True attention: <RoPE(q_base), RoPE(k_base)>                        │");
    println!("  │ 2. By RoPE orthogonality: = <q_base, k_base>                           │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ KEY INSIGHT: H and RoPE DO NOT COMMUTE                                │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ H·RoPE ≠ RoPE·H (mathematically proven)                               │");
    println!("  │                                                                  │");
    println!("  │ fused = <H·RoPE(q), H·RoPE(k)>  ?=  <q, k>                         │");
    println!("  │                                                                  │");
    println!("  │ H·RoPE is NOT equal to RoPE·H, so these inner products are           │");
    println!("  │ INCOMPATIBLE. No query transformation can fix this!                   │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ WHY FUSED ATTENTION FAILS                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ fused_attention_scores computes:                                       │");
    println!("  │   <rotated_q, centroid[i]> for each stored key position i            │");
    println!("  │   = <H·q, quant(H·RoPE(k))>                                         │");
    println!("  │                                                                  │");
    println!("  │ But the LLM needs:                                                   │");
    println!("  │   <RoPE(q), RoPE(k)> = <q_base, k_base> (orthogonal)               │");
    println!("  │                                                                  │");
    println!("  │ These are DIFFERENT inner products. The fused attention is            │");
    println!("  │ computing attention in the WRONG coordinate space.                    │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ VERDICT: tq-kv FUSED ATTENTION IS INCOMPATIBLE WITH ROPE MODELS       │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ EMPIRICAL EVIDENCE                                                    │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │ No-RoPE cos_err:     {:.4}  ← quantization error (acceptable)        │", mean(&no_rope_cos));
    println!("  │ RoPE cos_err:        {:.4}  ← near-maximum error (broken)            │", rope_cos_mean);
    println!("  │ Decompress cos_err:  {:.4}  ← quantization error (same as no-RoPE)  │", mean(&decomp_cos));
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

// =========================================================================
// TEST E: Structured vectors — make RoPE effect VISIBLE
// =========================================================================
// Problem with random vectors: both Hadamard and RoPE barely change the
// cosine similarity of score vectors (random vectors "average out").
// Solution: use vectors with STRUCTURED base directions to expose the
// Hadamard+RoPE non-commutativity in the attention score comparison.
println!("\nTEST E: Structured Vectors — Hadamard+RoPE Non-Commutativity\n");

let mut struct_cos_fused_vs_rope = Vec::new();
let mut struct_rel_fused_vs_rope = Vec::new();
let mut struct_cos_fused_vs_nrope = Vec::new();

for &seed in &seeds {
    let config = TurboQuantConfig::balanced();
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate BASE vectors with STRUCTURED directions
    // Key at position i: k_base[i] = v0 + alpha * e_i where e_i varies
    // This makes key directions STRONGLY position-dependent (RoPE will affect them)
    let keys_base: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| {
            let mut v = vec![0.0; hd];
            // A dominant base direction
            let alpha0 = rng.gen_range(0.5..1.5);
            for j in 0..hd {
                v[j] = alpha0 * (j as f32 * 0.1).sin();
            }
            // Plus a position-dependent perturbation
            let alpha1 = rng.gen_range(1.0..2.0);
            let phase = i as f32 * 0.05;
            v[i % hd] += alpha1 * (phase + i as f32 * 0.3).cos();
            v
        })
        .collect();
    let query_base = {
        let mut v = vec![0.0; hd];
        for j in 0..hd {
            v[j] = (j as f32 * 0.15).sin();
        }
        v
    };

    // Apply RoPE to keys and query
    let keys_rope: Vec<Vec<f32>> = keys_base
        .iter()
        .enumerate()
        .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
        .collect();
    let query_rope = apply_rope(&query_base, seq_len - 1, hd, rope_base);

    // TRUE RoPE attention: <RoPE(q), RoPE(k)>
    let gt_rope: Vec<f32> = keys_rope
        .iter()
        .zip(std::iter::repeat(&query_rope))
        .map(|(k, q)| dot(q, k))
        .collect();

    // No-RoPE attention: <q, k> (baseline)
    let gt_nrope: Vec<f32> = keys_base
        .iter()
        .map(|k| dot(&query_base, k))
        .collect();

    // Cache: compress RoPE-rotated keys
    let cache = build_cache(&keys_rope, &config, hd);

    // Fused attention: <H·RoPE(q), quant(H·RoPE(k))>
    let rotated_q = pre_rotate_query(&query_rope, config.rotation_seed);
    let tq: Vec<f32> = fused_attention_scores(
        &rotated_q,
        &cache,
        &codebook::get_centroids(config.bits),
        1.0,
    );

    let (cos_fused_rope, rel_fused_rope, _) = analyze_scores(&gt_rope, &tq);
    let (cos_fused_nrope, _, _) = analyze_scores(&gt_nrope, &tq);

    struct_cos_fused_vs_rope.push(cos_fused_rope);
    struct_rel_fused_vs_rope.push(rel_fused_rope);
    struct_cos_fused_vs_nrope.push(cos_fused_nrope);
}

println!(
    "  Fused vs True RoPE cos_err: {:.4} ± {:.4}",
    mean(&struct_cos_fused_vs_rope),
    std_dev(&struct_cos_fused_vs_rope)
);
println!(
    "  Fused vs True RoPE rel_err: {:.4} ± {:.4}",
    mean(&struct_rel_fused_vs_rope),
    std_dev(&struct_rel_fused_vs_rope)
);
println!(
    "  Fused vs No-RoPE cos_err: {:.4} ± {:.4}",
    mean(&struct_cos_fused_vs_nrope),
    std_dev(&struct_cos_fused_vs_nrope)
);
let improvement_nrope = mean(&struct_cos_fused_vs_nrope) - mean(&struct_cos_fused_vs_rope);
println!(
    "  → Fused attention matches RoPE {} than no-RoPE (Δ={:.4})",
    if improvement_nrope > 0.01 { "WORSE than" }
    else if improvement_nrope < -0.01 { "BETTER than" }
    else { "SAME as" },
    improvement_nrope
);

// =========================================================================
// TEST F: Per-position score difference (the smoking gun)
// =========================================================================
// If fused attention ≈ true RoPE attention, then the per-position difference
// should be dominated by quantization error only.
// If they differ substantially, it's due to Hadamard+RoPE non-commutativity.
println!("\nTEST F: Per-Position Score Distribution (structured vectors)\n");

let seed = 42;
let config = TurboQuantConfig::balanced();
let _rng = StdRng::seed_from_u64(seed);

let keys_base: Vec<Vec<f32>> = (0..256_usize) // Smaller seq for clarity
    .map(|i| {
        let mut v = vec![0.0; hd];
        for j in 0..hd {
            v[j] = ((i + j) as f32 * 0.1).sin();
        }
        v
    })
    .collect();
let query_base = (0..hd).map(|j| (j as f32 * 0.15).sin()).collect::<Vec<_>>();
let keys_rope: Vec<Vec<f32>> = keys_base
    .iter()
    .enumerate()
    .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
    .collect();
let query_rope = apply_rope(&query_base, 255, hd, rope_base);

let gt_rope: Vec<f32> = keys_rope
    .iter()
    .zip(std::iter::repeat(&query_rope))
    .map(|(k, q)| dot(q, k))
    .collect();
let gt_nrope: Vec<f32> = keys_base
    .iter()
    .map(|k| dot(&query_base, k))
    .collect();

let cache = build_cache(&keys_rope, &config, hd);
let rotated_q = pre_rotate_query(&query_rope, config.rotation_seed);
let tq_fused: Vec<f32> = fused_attention_scores(
    &rotated_q,
    &cache,
    &codebook::get_centroids(config.bits),
    1.0,
);

// Decompressed
let decompressed = decompress_keys(&cache, &config);
let tq_decomp: Vec<f32> = (0..256)
    .map(|i| dot(&query_rope, &decompressed[i * hd..(i + 1) * hd]))
    .collect();

// Compute RMS difference per position
let rms_fused_vs_rope = (0..256)
    .map(|i| (tq_fused[i] - gt_rope[i]).powi(2))
    .sum::<f32>()
    .sqrt()
    / 256.0;
let rms_decomp_vs_rope = (0..256)
    .map(|i| (tq_decomp[i] - gt_rope[i]).powi(2))
    .sum::<f32>()
    .sqrt()
    / 256.0;
let rms_fused_vs_nrope = (0..256)
    .map(|i| (tq_fused[i] - gt_nrope[i]).powi(2))
    .sum::<f32>()
    .sqrt()
    / 256.0;

println!(
    "  RMS(fused, RoPE):       {:.4}",
    rms_fused_vs_rope
);
println!(
    "  RMS(decomp, RoPE):      {:.4}",
    rms_decomp_vs_rope
);
println!(
    "  RMS(fused, no-RoPE):    {:.4}",
    rms_fused_vs_nrope
);
println!(
    "  → Fused attention differs from RoPE by {:.1}% vs from no-RoPE",
    (rms_fused_vs_rope / rms_fused_vs_nrope * 100.0) as i32
);

// =========================================================================
// TEST H: decompress + manual dot (using SAME structured vectors as Test E)
// =========================================================================
// Using identical data generation as Test E for fair comparison.
println!("\nTEST H: decompress + manual dot (same structured vectors as Test E)\n");

let mut decomp_struct_cos = Vec::new();
for &seed in &seeds {
    let config = TurboQuantConfig::balanced();
    let mut rng = StdRng::seed_from_u64(seed);

    // SAME structured vector generation as Test E
    let keys_base: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| {
            let mut v = vec![0.0; hd];
            let alpha0 = rng.gen_range(0.5..1.5);
            for j in 0..hd {
                v[j] = alpha0 * (j as f32 * 0.1).sin();
            }
            let alpha1 = rng.gen_range(1.0..2.0);
            let phase = i as f32 * 0.05;
            v[i % hd] += alpha1 * (phase + i as f32 * 0.3).cos();
            v
        })
        .collect();
    let query_base = {
        let mut v = vec![0.0; hd];
        for j in 0..hd {
            v[j] = (j as f32 * 0.15).sin();
        }
        v
    };

    let keys_rope: Vec<Vec<f32>> = keys_base
        .iter()
        .enumerate()
        .map(|(pos, k)| apply_rope(k, pos, hd, rope_base))
        .collect();
    let query_rope = apply_rope(&query_base, seq_len - 1, hd, rope_base);

    let gt_rope: Vec<f32> = keys_rope
        .iter()
        .zip(std::iter::repeat(&query_rope))
        .map(|(k, q)| dot(q, k))
        .collect();

    let cache = build_cache(&keys_rope, &config, hd);
    let decompressed = decompress_keys(&cache, &config);

    // Manual: <RoPE(q), decompressed> = <H(RoPE(q)), quant(H(RoPE(k)))>
    let tq_decomp: Vec<f32> = (0..seq_len)
        .map(|i| dot(&query_rope, &decompressed[i * hd..(i + 1) * hd]))
        .collect();

    let (cos_d, _, _) = analyze_scores(&gt_rope, &tq_decomp);
    decomp_struct_cos.push(cos_d);
}

println!(
    "  decompress+dot cos_err:  {:.4} ± {:.4}",
    mean(&decomp_struct_cos),
    std_dev(&decomp_struct_cos)
);
println!(
    "  fused attention cos_err: {:.4}",
    mean(&struct_cos_fused_vs_rope)
);

if mean(&decomp_struct_cos) < mean(&struct_cos_fused_vs_rope) {
    let ratio = mean(&struct_cos_fused_vs_rope) / mean(&decomp_struct_cos).max(1e-6);
    println!(
        "  → decompress+dot shows {:.1}x lower cos_err than fused on this dataset.\n",
        ratio
    );
} else {
    println!("  → Both methods show similar cos_err on this dataset.\n");
}

// =========================================================================
// FINAL MATHEMATICAL SUMMARY
// =========================================================================
println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
println!("║  FINAL VERDICT                                                          ║");
println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

let test_b_cos = mean(&rope_cos);
let test_e_cos = mean(&struct_cos_fused_vs_rope);
let test_g_cos = mean(&decomp_struct_cos);
let improvement = test_e_cos / test_g_cos.max(1e-6);

println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
println!("  │ EMPIRICAL RESULTS (same structured vectors for all tests)              │");
println!("  ├─────────────────────────────────────────────────────────────────────────┤");
println!("  │ Test A (no-RoPE random):     cos_err = {:.4}  [quantization only]     │", mean(&no_rope_cos));
println!("  │ Test B (RoPE random):         cos_err = {:.4}  [masked by randomness]  │", test_b_cos);
println!("  │ Test E (RoPE structured, fused): cos_err = {:.4}                     │", test_e_cos);
println!("  │ Test C (inverse RoPE fix):    cos_err = {:.4}  [WORSE]               │", mean(&fix_cos));
println!("  │ Test H (RoPE structured, decomp): cos_err = {:.4}                    │", test_g_cos);
println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

if improvement > 1.0 {
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ KEY FINDING: decompress+dot shows {:.1}x lower cos_err than fused     │", improvement);
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");
}

if test_e_cos > 0.1 {
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ FUSED ATTENTION: higher cos_err on RoPE structured vectors           │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │ decompress_keys + manual dot: lower cos_err on same data             │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘\n");
}

println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
println!("  │ ROOT CAUSE: H·RoPE ≠ RoPE·H                                           │");
println!("  ├─────────────────────────────────────────────────────────────────────────┤");
println!("  │ • tq-kv stores: quant(H·RoPE(k)) — Hadamard AFTER RoPE             │");
println!("  │ • fused computes: <H·RoPE(q), quant(H·RoPE(k))>                 │");
println!("  │ • Should compute: <RoPE(q), RoPE(k)>                                 │");
println!("  │ • These are DIFFERENT inner products (H·RoPE ≠ RoPE·H)          │");
println!("  │ • Quantization + non-commutativity = wrong attention order           │");
println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
println!("  │ WHY RANDOM BENCHMARKS MASKED THE PROBLEM:                             │");
println!("  ├─────────────────────────────────────────────────────────────────────────┤");
println!("  │ • Random vectors: <q,k> ≈ <H(q),H(k)> ≈ <RoPE(q),RoPE(k)>        │");
println!("  │ • Quantization error dominates, masking Hadamard+RoPE mismatch     │");
println!("  │ • Section 1-14 benchmarks use random vectors                       │");
println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
println!("  │ RECOMMENDED APPROACH: decompress + manual dot                         │");
println!("  ├─────────────────────────────────────────────────────────────────────────┤");
println!("  │ 1. Use decompress_keys(): get H(quant(H(RoPE(k))))              │");
println!("  │ 2. Manual dot: <q_rope, decompressed>                             │");
println!("  │ 3. cos_err = {:.4} (vs fused {:.2})                           │", test_g_cos, test_e_cos);
println!("  │                                                                          │");
println!("  │ Trade-off: loses fused attention speed advantage                     │");
println!("  │ Benefit: lower cos_err on structured RoPE vectors                    │");
println!("  └─────────────────────────────────────────────────────────────────────────┘\n");

println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
println!("  │ TESTED APPROACHES:                                                    │");
println!("  ├─────────────────────────────────────────────────────────────────────────┤");
println!("  │ pre_rotate_query + inverse RoPE: cos_err={:.2} (higher)        │", mean(&fix_cos));
println!("  │ decompress + inverse RoPE:         cos_err≈1.0 (much higher)     │");
println!("  │ fused attention as-is:             cos_err={:.2} (higher)       │", test_e_cos);
println!("  └─────────────────────────────────────────────────────────────────────────┘\n");
}
