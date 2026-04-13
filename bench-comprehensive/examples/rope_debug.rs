// Debug test to understand Hadamard + RoPE behavior
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tq_kv::{
    hadamard, compress_keys, decompress_keys, CompressedKeys, TurboQuantConfig,
};

fn apply_rope(vector: &[f32], position: usize, head_dim: usize, base: f64) -> Vec<f32> {
    let mut result = vector.to_vec();
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let angle = position as f64 / base.powi(-2 * i as i32 / head_dim as i32);
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        let idx0 = i;
        let idx1 = i + half_dim;
        let x0 = result[idx0] as f64;
        let x1 = result[idx1] as f64;
        result[idx0] = (x0 * cos_theta - x1 * sin_theta) as f32;
        result[idx1] = (x1 * cos_theta + x0 * sin_theta) as f32;
    }
    result
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    let hd = 8;
    let rope_base = 500000.0;
    let pos = 3;

    let mut rng = StdRng::seed_from_u64(42);
    let k_base: Vec<f32> = (0..hd).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q_base: Vec<f32> = (0..hd).map(|_| rng.gen_range(-1.0..1.0)).collect();

    println!("Base key:   {:?}", k_base);
    println!("Base query: {:?}", q_base);

    // Original attention
    let orig_attn = dot(&q_base, &k_base);
    println!("\nOriginal attention <q,k>: {:.4}", orig_attn);

    // No RoPE: Hadamard only
    let config = TurboQuantConfig::balanced();
    let mut k_h = k_base.clone();
    hadamard::randomized_hadamard(&mut k_h, config.rotation_seed);
    let mut q_h = q_base.clone();
    hadamard::randomized_hadamard(&mut q_h, config.rotation_seed);
    let no_rope_attn = dot(&q_h, &k_h);
    println!("No RoPE <H·q, H·k>: {:.4} (vs orig: {:.6})", no_rope_attn, (orig_attn - no_rope_attn).abs());

    // RoPE then Hadamard (CORRECT LLM order)
    let k_rope = apply_rope(&k_base, pos, hd, rope_base);
    let mut k_rope_h = k_rope.clone();
    hadamard::randomized_hadamard(&mut k_rope_h, config.rotation_seed);
    let q_rope = apply_rope(&q_base, pos, hd, rope_base);
    let mut q_rope_h = q_rope.clone();
    hadamard::randomized_hadamard(&mut q_rope_h, config.rotation_seed);

    // True RoPE attention
    let roe_attn = dot(&q_rope, &k_rope);
    println!("True RoPE <RoPE(q),RoPE(k)>: {:.4} (vs orig: {:.6})", roe_attn, (orig_attn - roe_attn).abs());

    // Compress then decompress: what does decompress_keys return?
    let compressed = compress_keys(&k_rope, hd, &config);
    let decompressed = decompress_keys(&compressed, &config);
    println!("\nCompressed form H·RoPE(k): {:?}", &k_rope_h);
    println!("Decompressed keys:        {:?}", &decompressed[..hd]);
    println!("Is H·decompressed == H·RoPE(k)?");
    let mut h_decomp = decompressed[..hd].to_vec();
    hadamard::randomized_hadamard(&mut h_decomp, config.rotation_seed);
    println!("  H·decompressed: {:?}", &h_decomp);
    println!("  H·RoPE(k):      {:?}", &k_rope_h);

    // fused_attention: <H·RoPE(q), decompressed>
    let fused = dot(&q_rope_h, &decompressed[..hd]);
    println!("\nFused attention <H·RoPE(q), decompressed>: {:.4}", fused);
    println!("True RoPE attention: {:.4}", roe_attn);
    println!("Error (fused vs true RoPE): {:.6}", (roe_attn - fused).abs());
    println!("Error (fused vs original): {:.6}", (orig_attn - fused).abs());
}
