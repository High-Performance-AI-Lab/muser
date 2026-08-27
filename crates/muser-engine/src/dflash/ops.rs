#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

pub(crate) fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(out.len() >= m * n);
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = (0..k).map(|i| a[row * k + i] * b[col * k + i]).sum();
        }
    }
}

pub(crate) fn rms_norm(x: &mut [f32], weight: &[f32], rows: usize, dim: usize, eps: f64) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    for row in x.chunks_exact_mut(dim) {
        let inv = 1.
            / ((row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim as f64 + eps)
                .sqrt());
        for (value, weight) in row.iter_mut().zip(weight) {
            *value = (*value as f64 * inv * *weight as f64) as f32;
        }
    }
}

pub(crate) fn head_norm(
    x: &mut [f32],
    weight: &[f32],
    seq: usize,
    heads: usize,
    dim: usize,
    eps: f64,
) {
    for row in x.chunks_exact_mut(dim).take(seq * heads) {
        let inv = 1.
            / ((row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim as f64 + eps)
                .sqrt());
        for (value, weight) in row.iter_mut().zip(weight) {
            *value = (*value as f64 * inv * *weight as f64) as f32;
        }
    }
}

pub(crate) fn rope(x: &mut [f32], seq: usize, heads: usize, dim: usize, start: usize, theta: f64) {
    let half = dim / 2;
    for pos in 0..seq {
        for head in 0..heads {
            let base = (pos * heads + head) * dim;
            for i in 0..half {
                let angle = (start + pos) as f64 / theta.powf(2. * i as f64 / dim as f64);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (x[base + i], x[base + half + i]);
                x[base + i] = a * cos as f32 - b * sin as f32;
                x[base + half + i] = b * cos as f32 + a * sin as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) {
    let groups = heads / kv_heads;
    let scale = 1. / (dim as f32).sqrt();
    for head in 0..heads {
        let kv_head = head / groups;
        for qi in 0..q_seq {
            let mut scores = vec![0.; kv_seq];
            let qo = (qi * heads + head) * dim;
            for (ki, score) in scores.iter_mut().enumerate() {
                let ko = (ki * kv_heads + kv_head) * dim;
                *score = (0..dim).map(|d| q[qo + d] * k[ko + d]).sum::<f32>() * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max).exp();
                    *s
                })
                .sum::<f32>();
            let oo = (qi * heads + head) * dim;
            for d in 0..dim {
                out[oo + d] = (0..kv_seq)
                    .map(|ki| scores[ki] / sum * v[(ki * kv_heads + kv_head) * dim + d])
                    .sum();
            }
        }
    }
}
