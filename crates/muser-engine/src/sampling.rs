//! Canonical scalar sampling and full-distribution speculative acceptance.
//! The ordering follows Ferrite's request sampler; the rejection branch uses
//! the complete normalized max(target - draft, 0) distribution.

use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Domain separator for the finite, reproducible implementation of the
/// communication-free Gumbel coupling from Daliri, Musco, and Suresh.
///
/// The random field is indexed by absolute output position rather than by a
/// speculative round.  Changing the drafter, verification width, or round
/// boundaries therefore cannot change the target sampler's random numbers.
pub const SHARED_GUMBEL_DOMAIN_V1: &[u8] = b"muser-shared-gumbel-v1\0";
/// The scalar SHA/libm implementation is a correctness reference for bounded
/// support, not a full-vocabulary serving kernel.
pub const MAX_SHARED_GUMBEL_REFERENCE_SUPPORT: usize = 4_096;

/// Public randomness for drafter-invariant speculative sampling.
///
/// This is deliberately not an authentication key.  A verifier protocol must
/// bind these bytes, the sampler policy, and the absolute output position in
/// its authenticated session genesis.  Keeping the seed as 256 bits avoids
/// silently truncating a protocol nonce to the legacy request sampler's u32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedGumbelSeed([u8; 32]);

impl SharedGumbelSeed {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn from_u64(seed: u64) -> Self {
        let mut hash = Sha256::new();
        hash.update(SHARED_GUMBEL_DOMAIN_V1);
        hash.update(b"seed\0");
        hash.update(seed.to_be_bytes());
        Self(hash.finalize().into())
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mt19937Snapshot {
    pub state: Vec<u32>,
    pub index: usize,
}

/// Bit-for-bit implementation of the `std::mt19937` engine used by the
/// source-pinned llama.cpp sampler. Keeping it here (rather than using
/// `StdRng`, whose algorithm is deliberately unspecified) makes seeded API
/// results stable across Rust and `rand` releases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for index in 1..state.len() {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index == self.state.len() {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    pub fn snapshot(&self) -> Mt19937Snapshot {
        Mt19937Snapshot {
            state: self.state.to_vec(),
            index: self.index,
        }
    }

    pub fn from_snapshot(snapshot: &Mt19937Snapshot) -> Result<Self, SamplingError> {
        if snapshot.state.len() != 624 || snapshot.index > 624 {
            return Err(SamplingError::InvalidRngState);
        }
        let mut state = [0; 624];
        state.copy_from_slice(&snapshot.state);
        Ok(Self {
            state,
            index: snapshot.index,
        })
    }

    /// libc++'s `uniform_real_distribution<float>(0, 1)`: one 32-bit engine
    /// result scaled by 2^-32.
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() as f64 * (1.0 / 4_294_967_296.0)) as f32
    }

    /// libc++'s `uniform_real_distribution<double>(0, 1)` over mt19937.
    /// `generate_canonical<double, 53>` consumes the first result as the low
    /// limb and the second as the high limb.
    pub fn uniform_f64(&mut self) -> f64 {
        let low = self.next_u32() as f64;
        let high = self.next_u32() as f64;
        (low + high * 4_294_967_296.0) * (1.0 / 18_446_744_073_709_551_616.0)
    }

    fn twist(&mut self) {
        for index in 0..self.state.len() {
            let joined = (self.state[index] & 0x8000_0000)
                | (self.state[(index + 1) % self.state.len()] & 0x7fff_ffff);
            self.state[index] = self.state[(index + 397) % self.state.len()]
                ^ (joined >> 1)
                ^ if joined & 1 == 0 { 0 } else { 0x9908_b0df };
        }
        self.index = 0;
    }
}

/// Select in candidate order with the same double-precision uniform draw as
/// llama.cpp's scalar distribution sampler. Probabilities need not already
/// sum to one.
pub fn sample_distribution_mt(
    probabilities: &[f32],
    rng: &mut Mt19937,
) -> Result<u32, SamplingError> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = probabilities.iter().map(|value| *value as f64).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let target = rng.uniform_f64() * total;
    let mut cumulative = 0.0f64;
    for (token, probability) in probabilities.iter().enumerate() {
        cumulative += *probability as f64;
        if cumulative >= target {
            return Ok(token as u32);
        }
    }
    Ok(probabilities
        .iter()
        .enumerate()
        .rfind(|(_, probability)| **probability > 0.0)
        .map_or(0, |(token, _)| token as u32))
}

/// Select using libc++'s `std::discrete_distribution<int>` construction and
/// draw semantics. Mirostat and adaptive-p normalize each candidate into a
/// double-precision prefix table and select with `upper_bound`.
pub fn sample_discrete_distribution_mt(
    probabilities: &[f32],
    rng: &mut Mt19937,
) -> Result<u32, SamplingError> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = probabilities.iter().map(|value| *value as f64).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let draw = rng.uniform_f64();
    let mut cumulative = 0.0f64;
    for (token, probability) in probabilities
        .iter()
        .take(probabilities.len().saturating_sub(1))
        .enumerate()
    {
        cumulative += *probability as f64 / total;
        if draw < cumulative {
            return Ok(token as u32);
        }
    }
    Ok((probabilities.len() - 1) as u32)
}

pub fn sample_discrete_distribution_mt_ordered(
    probabilities: &[f32],
    order: &[u32],
    rng: &mut Mt19937,
) -> Result<u32, SamplingError> {
    if probabilities.is_empty()
        || order.is_empty()
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || order
            .iter()
            .any(|token| *token as usize >= probabilities.len())
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = order
        .iter()
        .map(|token| probabilities[*token as usize] as f64)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let draw = rng.uniform_f64();
    let mut cumulative = 0.0f64;
    for &token in order.iter().take(order.len().saturating_sub(1)) {
        cumulative += probabilities[token as usize] as f64 / total;
        if draw < cumulative {
            return Ok(token);
        }
    }
    Ok(*order.last().expect("order is nonempty"))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// Zero disables top-k.
    pub top_k: usize,
    pub typical_p: f32,
    pub min_p: f32,
    /// Keep logits no further than N standard deviations below the maximum;
    /// zero disables the filter.
    pub top_n_sigma: f32,
    pub min_keep: usize,
}

impl SamplingParams {
    pub fn validate(self) -> Result<(), SamplingError> {
        if !(self.temperature.is_finite()
            && self.temperature >= 0.0
            && self.top_p.is_finite()
            && 0.0 <= self.top_p
            && self.top_p <= 1.0
            && self.typical_p.is_finite()
            && 0.0 < self.typical_p
            && self.typical_p <= 1.0
            && self.min_p.is_finite()
            && 0.0 <= self.min_p
            && self.min_p <= 1.0
            && self.top_n_sigma.is_finite()
            && self.top_n_sigma >= 0.0)
        {
            return Err(SamplingError::InvalidParams);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SamplingError {
    #[error("sampling parameters are outside their finite canonical ranges")]
    InvalidParams,
    #[error("cannot sample empty or nonfinite logits")]
    InvalidLogits,
    #[error("invalid MT19937 snapshot")]
    InvalidRngState,
    #[error("draft/target speculative probability geometry differs")]
    Geometry,
}

/// Exact full-vocabulary distribution for the source-pinned llama.cpp
/// default chain: top-n-sigma, top-k, typical-p, top-p, min-p, then
/// temperature and distribution sampling. Penalties/DRY and XTC are applied
/// by the request sampler around this scalar core. Ties resolve by token ID.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedDistribution {
    /// Full-vocabulary normalized probabilities for logprobs/speculative math.
    pub probabilities: Vec<f32>,
    /// Full-vocabulary unnormalized expf weights as consumed by llama's final
    /// scalar distribution sampler.
    pub weights: Vec<f32>,
    /// Source candidate order after the configured truncation chain.
    pub order: Vec<u32>,
}

/// Wire-friendly support of a normalized, sampler-ordered distribution.
///
/// Muser's default sampled route applies top-k=40.  Sending only these
/// positive entries makes ordinary maximal-coupling verification exact for
/// that already-truncated product distribution without moving a 202,048-wide
/// logit row.  `entries` retain the source sampler's candidate order because
/// fixed-seed selection is observably order-sensitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseProbabilityRow {
    pub vocab_size: u32,
    pub entries: Vec<SparseProbabilityEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparseProbabilityEntry {
    pub token: u32,
    pub probability: f32,
}

impl SparseProbabilityRow {
    pub fn from_ordered(distribution: &OrderedDistribution) -> Result<Self, SamplingError> {
        let vocab_size =
            u32::try_from(distribution.probabilities.len()).map_err(|_| SamplingError::Geometry)?;
        let mut entries = Vec::with_capacity(distribution.order.len());
        let mut seen = vec![false; distribution.probabilities.len()];
        for &token in &distribution.order {
            let index = token as usize;
            let probability = *distribution
                .probabilities
                .get(index)
                .ok_or(SamplingError::Geometry)?;
            if seen[index] {
                return Err(SamplingError::Geometry);
            }
            seen[index] = true;
            if probability > 0.0 {
                entries.push(SparseProbabilityEntry { token, probability });
            }
        }
        if distribution
            .probabilities
            .iter()
            .zip(seen)
            .any(|(probability, present)| *probability > 0.0 && !present)
        {
            return Err(SamplingError::Geometry);
        }
        let row = Self {
            vocab_size,
            entries,
        };
        row.validate()?;
        Ok(row)
    }

    pub fn validate(&self) -> Result<(), SamplingError> {
        if self.vocab_size == 0 || self.entries.is_empty() {
            return Err(SamplingError::InvalidLogits);
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0.0f64;
        for entry in &self.entries {
            if entry.token >= self.vocab_size
                || !seen.insert(entry.token)
                || !entry.probability.is_finite()
                || entry.probability <= 0.0
            {
                return Err(SamplingError::InvalidLogits);
            }
            total += entry.probability as f64;
        }
        // Entries are normalized f32 probabilities, not arbitrary weights.
        // Each value was rounded independently after a common f64 total; the
        // support-sized envelope admits that rounding and rejects a scaled q
        // row that would alter p/q acceptance.
        let tolerance = 1e-6;
        if !total.is_finite() || (total - 1.0).abs() > tolerance {
            return Err(SamplingError::InvalidLogits);
        }
        Ok(())
    }

    pub fn validate_bounded(&self, max_support: usize) -> Result<(), SamplingError> {
        self.validate()?;
        if max_support == 0 || self.entries.len() > max_support {
            return Err(SamplingError::Geometry);
        }
        Ok(())
    }

    pub fn probability(&self, token: u32) -> f32 {
        self.entries
            .iter()
            .find(|entry| entry.token == token)
            .map_or(0.0, |entry| entry.probability)
    }
}

pub fn distribution(logits: &[f32], params: SamplingParams) -> Result<Vec<f32>, SamplingError> {
    Ok(distribution_ordered(logits, params)?.probabilities)
}

pub fn distribution_ordered(
    logits: &[f32],
    params: SamplingParams,
) -> Result<OrderedDistribution, SamplingError> {
    params.validate()?;
    if logits.is_empty()
        || logits
            .iter()
            .any(|value| value.is_nan() || *value == f32::INFINITY)
        || logits.iter().all(|value| *value == f32::NEG_INFINITY)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let mut candidates = logits
        .iter()
        .enumerate()
        .map(|(token, &logit)| (token, logit))
        .collect::<Vec<_>>();
    let mut sorted = false;
    if params.top_n_sigma > 0.0 {
        let finite = logits
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = finite.iter().copied().sum::<f32>() / finite.len() as f32;
        let variance = finite
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>()
            / finite.len() as f32;
        let threshold = max - params.top_n_sigma * variance.sqrt();
        // Upstream masks in place and intentionally leaves candidate order
        // untouched. Zero-weight entries remain observable to later samplers.
        for (_, logit) in &mut candidates {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    if params.top_k > 0 {
        sort_candidates_descending(&mut candidates);
        candidates.truncate(params.top_k.min(candidates.len()));
        sorted = true;
    }
    if params.typical_p < 1.0 {
        if !sorted {
            sort_candidates_descending(&mut candidates);
        }
        let probabilities = unit_temperature_probabilities(&candidates)?;
        let entropy = probabilities
            .iter()
            .map(|&probability| -probability * probability.ln())
            .sum::<f32>();
        let mut typical = candidates
            .into_iter()
            .zip(probabilities)
            .collect::<Vec<_>>();
        typical.sort_unstable_by(|left, right| {
            let left_probability = left.1;
            let right_probability = right.1;
            (-left_probability.ln() - entropy)
                .abs()
                .total_cmp(&(-right_probability.ln() - entropy).abs())
                .then_with(|| left.0 .0.cmp(&right.0 .0))
        });
        let mut cumulative = 0.0;
        let mut keep = typical.len();
        for (index, (_, probability)) in typical.iter().enumerate() {
            cumulative += probability;
            if cumulative > params.typical_p && index + 1 >= params.min_keep {
                keep = index + 1;
                break;
            }
        }
        typical.truncate(keep.max(1));
        candidates = typical.into_iter().map(|(entry, _)| entry).collect();
        // Locally-typical order is part of the source contract; a later
        // top-p sampler may sort it again, but typical alone must not.
        sorted = false;
    }
    if params.top_p < 1.0 {
        let probabilities = unit_temperature_probabilities(&candidates)?;
        if !sorted {
            let mut with_probabilities = candidates
                .into_iter()
                .zip(probabilities)
                .collect::<Vec<_>>();
            with_probabilities.sort_unstable_by(|left, right| {
                right
                    .0
                     .1
                    .total_cmp(&left.0 .1)
                    .then_with(|| left.0 .0.cmp(&right.0 .0))
            });
            candidates = with_probabilities
                .iter()
                .map(|(candidate, _)| *candidate)
                .collect();
        }
        let probabilities = unit_temperature_probabilities(&candidates)?;
        let mut cumulative = 0.0f32;
        let mut cutoff = candidates.len();
        for (index, probability) in probabilities.into_iter().enumerate() {
            cumulative += probability;
            if cumulative >= params.top_p && index + 1 >= params.min_keep {
                cutoff = index + 1;
                break;
            }
        }
        candidates.truncate(cutoff.max(1));
        sorted = true;
    }
    if params.min_p > 0.0 {
        let maximum = candidates
            .iter()
            .map(|entry| entry.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = maximum + params.min_p.ln();
        if !sorted {
            let filtered = candidates
                .iter()
                .copied()
                .filter(|entry| entry.1 >= threshold)
                .collect::<Vec<_>>();
            if !filtered.is_empty() && filtered.len() >= params.min_keep {
                candidates = filtered;
            } else {
                sort_candidates_descending(&mut candidates);
                sorted = true;
            }
        }
        if sorted {
            let matching = candidates
                .iter()
                .take_while(|entry| entry.1 >= threshold)
                .count();
            candidates.truncate(matching.max(params.min_keep).max(1).min(candidates.len()));
        }
    }
    if candidates.is_empty() {
        return Err(SamplingError::InvalidLogits);
    }
    // llama_sampler_temp_impl performs an f32 division before dist computes
    // expf(logit - max). Preserve that rounding boundary.
    let scaled = if params.temperature <= 0.0 {
        let mut maximum_index = 0usize;
        for index in 1..candidates.len() {
            if candidates[index].1 > candidates[maximum_index].1 {
                maximum_index = index;
            }
        }
        candidates
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index == maximum_index {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>()
    } else {
        candidates
            .iter()
            .map(|entry| entry.1 / params.temperature)
            .collect::<Vec<_>>()
    };
    let maximum = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weighted = scaled
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|value| *value as f64).sum::<f64>();
    let order = candidates
        .iter()
        .map(|(token, _)| *token as u32)
        .collect::<Vec<_>>();
    let mut probabilities = vec![0.0f32; logits.len()];
    let mut weights = vec![0.0f32; logits.len()];
    for ((token, _), weight) in candidates.into_iter().zip(weighted) {
        weights[token] = weight;
        probabilities[token] = (weight as f64 / total) as f32;
    }
    Ok(OrderedDistribution {
        probabilities,
        weights,
        order,
    })
}

fn sort_candidates_descending(candidates: &mut [(usize, f32)]) {
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

/// Sample full-vocabulary weights in the candidate order retained by the
/// source chain. Iterating token IDs is observably different under a fixed
/// seed even when the probability assigned to every token is identical.
pub fn sample_distribution_mt_ordered(
    weights: &[f32],
    order: &[u32],
    rng: &mut Mt19937,
) -> Result<u32, SamplingError> {
    if weights.is_empty()
        || order.is_empty()
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || order.iter().any(|token| *token as usize >= weights.len())
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = order
        .iter()
        .map(|token| weights[*token as usize] as f64)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let target = rng.uniform_f64() * total;
    let mut cumulative = 0.0f64;
    for &token in order {
        cumulative += weights[token as usize] as f64;
        if cumulative >= target {
            return Ok(token);
        }
    }
    Ok(*order.last().expect("order is nonempty"))
}

fn unit_temperature_probabilities(ranked: &[(usize, f32)]) -> Result<Vec<f32>, SamplingError> {
    let maximum = ranked
        .iter()
        .map(|entry| entry.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights = ranked
        .iter()
        .map(|entry| (entry.1 - maximum).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    Ok(weights.into_iter().map(|value| value / total).collect())
}

pub fn sample_distribution(
    probabilities: &[f32],
    rng: &mut impl Rng,
) -> Result<u32, SamplingError> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = probabilities.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let mut threshold = rng.gen::<f32>() * total;
    for (token, &probability) in probabilities.iter().enumerate() {
        threshold -= probability;
        if threshold <= 0.0 {
            return Ok(token as u32);
        }
    }
    Ok(probabilities
        .iter()
        .enumerate()
        .rfind(|(_, probability)| **probability > 0.0)
        .map(|(token, _)| token as u32)
        .unwrap_or(0))
}

/// Draw from locally transformed weights using the same public Gumbel field
/// as a remote drafter or target.
///
/// `output_position` is the absolute index in the generated response, not a
/// position inside the current speculative block.  The caller must use the
/// same seed and position on both endpoints.  Weights need not be normalized;
/// zero-weight tokens are excluded.  Ties resolve by token ID so candidate
/// iteration order is not part of the distributed contract.
///
/// The SHA-256 output is reduced to an open-interval 52-bit uniform before the
/// Gumbel transform.  This avoids the `ln(0)`/`ln(1)` endpoints and makes the
/// host reference stable without pretending that a finite PRF is an ideal
/// real-valued random oracle. This scalar SHA/libm implementation is capped at
/// [`MAX_SHARED_GUMBEL_REFERENCE_SUPPORT`]. A serving GPU implementation must
/// bind its implementation digest and cross-endpoint test vectors; different
/// target transcendental rounding is a new sampler epoch, not replay-equivalent
/// failover. Draft-side drift may raise or lower collisions, but cannot change
/// the output selected by the sole target authority.
pub fn sample_shared_gumbel_ordered(
    weights: &[f32],
    order: &[u32],
    seed: SharedGumbelSeed,
    output_position: u64,
) -> Result<u32, SamplingError> {
    if weights.is_empty()
        || order.is_empty()
        || order.len() > MAX_SHARED_GUMBEL_REFERENCE_SUPPORT
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let mut seen = vec![false; weights.len()];
    let mut winner = None::<(u32, f64)>;
    for &token in order {
        let index = token as usize;
        if index >= weights.len() || seen[index] {
            return Err(SamplingError::InvalidLogits);
        }
        seen[index] = true;
        let weight = weights[index];
        if weight == 0.0 {
            continue;
        }
        let uniform = shared_gumbel_uniform(seed, output_position, token);
        let gumbel = -(-uniform.ln()).ln();
        let score = (weight as f64).ln() + gumbel;
        match winner {
            None => winner = Some((token, score)),
            Some((winner_token, winner_score))
                if score > winner_score || (score == winner_score && token < winner_token) =>
            {
                winner = Some((token, score));
            }
            _ => {}
        }
    }
    if weights
        .iter()
        .enumerate()
        .any(|(index, weight)| *weight > 0.0 && !seen[index])
    {
        return Err(SamplingError::InvalidLogits);
    }
    winner
        .map(|(token, _)| token)
        .ok_or(SamplingError::InvalidLogits)
}

fn shared_gumbel_uniform(seed: SharedGumbelSeed, output_position: u64, token: u32) -> f64 {
    let mut hash = Sha256::new();
    hash.update(SHARED_GUMBEL_DOMAIN_V1);
    hash.update(seed.as_bytes());
    hash.update(output_position.to_be_bytes());
    hash.update(token.to_be_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    let raw = u64::from_be_bytes(digest[..8].try_into().expect("eight-byte SHA prefix"));
    open_unit_interval_from_u64(raw)
}

fn open_unit_interval_from_u64(raw: u64) -> f64 {
    // A binary64 has enough precision to represent every half-integer below
    // 2^52 exactly. Using 53 source bits here is subtly wrong: at the maximum,
    // `(2^53 - 1) + 0.5` rounds to 2^53 and produces the forbidden endpoint
    // U=1. The 52-bit midpoint map is exactly in (0, 1).
    let mantissa = raw >> 12;
    (mantissa as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeDecision {
    pub accepted: usize,
    pub next_token: u32,
}

/// One decision in Muser's carried-frontier speculative state machine.
///
/// `frontier_out` has been selected by the target sampler but has not been
/// evaluated, entered into KV, emitted, or counted as committed.  It becomes
/// `frontier_in` (candidate zero) on the next round.  Keeping this state
/// explicit prevents a rejection token from being published before the target
/// has processed its row and captured the corresponding DFlash features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterInvariantFrontierDecision {
    pub accepted_drafts: usize,
    pub commit_input_count: usize,
    pub frontier_out: u32,
}

/// Decide one drafter-invariant block in Muser's carried-frontier geometry.
///
/// `target_samples` contains a durable witness for `frontier_in`, followed by
/// one target sample after evaluating the frontier and each successive draft.
/// It therefore has `draft_tokens.len() + 2` entries.  Draft `i` is compared
/// with target sample `i + 1`; the matching/evaluated inputs are committed and
/// the first mismatch (or all-matched bonus) remains the next unprocessed
/// frontier.  Random fields used to produce those samples must be indexed by
/// absolute generated-token ordinal, never by speculative round boundaries.
pub fn verify_drafter_invariant_frontier(
    frontier_in: u32,
    draft_tokens: &[u32],
    target_samples: &[u32],
) -> Result<DrafterInvariantFrontierDecision, SamplingError> {
    if target_samples.len() != draft_tokens.len() + 2 || target_samples[0] != frontier_in {
        return Err(SamplingError::Geometry);
    }
    for (index, (&draft, &target)) in draft_tokens.iter().zip(&target_samples[1..]).enumerate() {
        if draft != target {
            return Ok(DrafterInvariantFrontierDecision {
                accepted_drafts: index,
                commit_input_count: index + 1,
                frontier_out: target,
            });
        }
    }
    Ok(DrafterInvariantFrontierDecision {
        accepted_drafts: draft_tokens.len(),
        commit_input_count: draft_tokens.len() + 1,
        frontier_out: *target_samples.last().ok_or(SamplingError::Geometry)?,
    })
}

/// Scalar Leviathan-style acceptance over full p/q distributions. Target
/// distributions contain one row per draft plus the all-accepted boundary.
pub fn verify_full_speculative(
    draft_tokens: &[u32],
    draft_probabilities: &[Vec<f32>],
    target_probabilities: &[Vec<f32>],
    rng: &mut impl Rng,
) -> Result<SpeculativeDecision, SamplingError> {
    if draft_tokens.len() != draft_probabilities.len()
        || target_probabilities.len() != draft_tokens.len() + 1
        || draft_probabilities
            .iter()
            .chain(target_probabilities)
            .any(|row| row.len() != target_probabilities[0].len())
    {
        return Err(SamplingError::Geometry);
    }
    for (index, (&token, (draft, target))) in draft_tokens
        .iter()
        .zip(draft_probabilities.iter().zip(target_probabilities))
        .enumerate()
    {
        let token = token as usize;
        if token >= draft.len() {
            return Err(SamplingError::Geometry);
        }
        let q = draft[token];
        let p = target[token];
        let acceptance = if q <= 0.0 { 1.0 } else { (p / q).min(1.0) };
        if rng.gen::<f32>() <= acceptance {
            continue;
        }
        let mut residual = target
            .iter()
            .zip(draft)
            .map(|(&p, &q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        let total = residual.iter().sum::<f32>();
        if total <= 0.0 {
            residual.clone_from(target);
        } else {
            for probability in &mut residual {
                *probability /= total;
            }
        }
        return Ok(SpeculativeDecision {
            accepted: index,
            next_token: sample_distribution(&residual, rng)?,
        });
    }
    Ok(SpeculativeDecision {
        accepted: draft_tokens.len(),
        next_token: sample_distribution(
            target_probabilities.last().ok_or(SamplingError::Geometry)?,
            rng,
        )?,
    })
}

/// Maximal-coupling speculative verification over sparse normalized rows.
///
/// This is distribution-equivalent to [`verify_full_speculative_mt_ordered`]
/// when every row contains exactly its positive support in the same candidate
/// order.  It is intended for the bounded top-k distributed lane: a draft can
/// send at most K `(token, probability)` pairs per position, while the target
/// keeps its own sparse rows local.  The rejection residual has support only
/// where target probability is positive, so no omitted vocabulary entry is
/// needed to sample `max(p - q, 0)`.
pub fn verify_sparse_speculative_mt(
    draft_tokens: &[u32],
    draft_rows: &[SparseProbabilityRow],
    target_rows: &[SparseProbabilityRow],
    max_support: usize,
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, SamplingError> {
    if max_support == 0
        || draft_tokens.len() != draft_rows.len()
        || target_rows.len() != draft_tokens.len() + 1
        || target_rows.is_empty()
    {
        return Err(SamplingError::Geometry);
    }
    let vocab_size = target_rows[0].vocab_size;
    for row in draft_rows.iter().chain(target_rows) {
        row.validate_bounded(max_support)?;
        if row.vocab_size != vocab_size {
            return Err(SamplingError::Geometry);
        }
    }

    for (index, (&token, (draft, target))) in draft_tokens
        .iter()
        .zip(draft_rows.iter().zip(target_rows))
        .enumerate()
    {
        if token >= vocab_size {
            return Err(SamplingError::Geometry);
        }
        let q = draft.probability(token);
        if q <= 0.0 {
            return Err(SamplingError::Geometry);
        }
        let p = target.probability(token);
        let acceptance = (p / q).min(1.0);
        if rng.uniform_f32() < acceptance {
            continue;
        }

        let mut residual = target
            .entries
            .iter()
            .map(|entry| SparseProbabilityEntry {
                token: entry.token,
                probability: (entry.probability - draft.probability(entry.token)).max(0.0),
            })
            .filter(|entry| entry.probability > 0.0)
            .collect::<Vec<_>>();
        // The dense reference sums the full residual in token-ID order before
        // sampling in the target candidate order.  Preserve that f32 rounding
        // boundary so sparse transport does not change seeded API tokens.
        let mut token_order = residual.clone();
        token_order.sort_unstable_by_key(|entry| entry.token);
        let total = token_order
            .iter()
            .map(|entry| entry.probability)
            .sum::<f32>();
        if total <= 0.0 {
            residual.clone_from(&target.entries);
        } else {
            for entry in &mut residual {
                entry.probability /= total;
            }
        }
        return Ok(SpeculativeDecision {
            accepted: index,
            next_token: sample_sparse_probability_row_mt(&residual, rng)?,
        });
    }
    Ok(SpeculativeDecision {
        accepted: draft_tokens.len(),
        next_token: sample_sparse_probability_row_mt(
            &target_rows.last().ok_or(SamplingError::Geometry)?.entries,
            rng,
        )?,
    })
}

fn sample_sparse_probability_row_mt(
    entries: &[SparseProbabilityEntry],
    rng: &mut Mt19937,
) -> Result<u32, SamplingError> {
    if entries.is_empty()
        || entries
            .iter()
            .any(|entry| !entry.probability.is_finite() || entry.probability <= 0.0)
    {
        return Err(SamplingError::InvalidLogits);
    }
    let total = entries
        .iter()
        .map(|entry| entry.probability as f64)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::InvalidLogits);
    }
    let target = rng.uniform_f64() * total;
    let mut cumulative = 0.0f64;
    for entry in entries {
        cumulative += entry.probability as f64;
        if cumulative >= target {
            return Ok(entry.token);
        }
    }
    Ok(entries.last().expect("entries are nonempty").token)
}

/// The same full-distribution speculative decision using the exact
/// source-pinned `std::mt19937` draw stream. Acceptance consumes one
/// `uniform_real_distribution<float>` draw per attempted draft and token
/// selection consumes the libc++ double-precision discrete draw used by the
/// ordinary request sampler. Keeping this separate from the generic `rand`
/// helper prevents a `rand` algorithm or conversion change from altering API
/// tokens or a persisted session frontier.
pub fn verify_full_speculative_mt(
    draft_tokens: &[u32],
    draft_probabilities: &[Vec<f32>],
    target_probabilities: &[Vec<f32>],
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, SamplingError> {
    let target_orders = target_probabilities
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, probability)| **probability > 0.0)
                .map(|(token, _)| token as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    verify_full_speculative_mt_ordered(
        draft_tokens,
        draft_probabilities,
        target_probabilities,
        &target_orders,
        rng,
    )
}

pub fn verify_full_speculative_mt_ordered(
    draft_tokens: &[u32],
    draft_probabilities: &[Vec<f32>],
    target_probabilities: &[Vec<f32>],
    target_orders: &[Vec<u32>],
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, SamplingError> {
    if draft_tokens.len() != draft_probabilities.len()
        || target_probabilities.len() != draft_tokens.len() + 1
        || target_orders.len() != target_probabilities.len()
        || target_probabilities.is_empty()
        || draft_probabilities
            .iter()
            .chain(target_probabilities)
            .any(|row| row.len() != target_probabilities[0].len())
    {
        return Err(SamplingError::Geometry);
    }
    for (index, (&token, (draft, target))) in draft_tokens
        .iter()
        .zip(draft_probabilities.iter().zip(target_probabilities))
        .enumerate()
    {
        let token = token as usize;
        if token >= draft.len() {
            return Err(SamplingError::Geometry);
        }
        let q = draft[token];
        let p = target[token];
        let acceptance = if q <= 0.0 { 1.0 } else { (p / q).min(1.0) };
        if rng.uniform_f32() <= acceptance {
            continue;
        }
        let mut residual = target
            .iter()
            .zip(draft)
            .map(|(&p, &q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        let total = residual.iter().sum::<f32>();
        if total <= 0.0 {
            residual.clone_from(target);
        } else {
            for probability in &mut residual {
                *probability /= total;
            }
        }
        let order = target_orders[index]
            .iter()
            .copied()
            .filter(|token| residual[*token as usize] > 0.0)
            .collect::<Vec<_>>();
        return Ok(SpeculativeDecision {
            accepted: index,
            next_token: sample_distribution_mt_ordered(&residual, &order, rng)?,
        });
    }
    Ok(SpeculativeDecision {
        accepted: draft_tokens.len(),
        next_token: sample_distribution_mt_ordered(
            target_probabilities.last().ok_or(SamplingError::Geometry)?,
            target_orders.last().ok_or(SamplingError::Geometry)?,
            rng,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn mt19937_matches_libcxx_engine_and_uniform_distributions() {
        let mut raw = Mt19937::new(42);
        assert_eq!(
            (0..6).map(|_| raw.next_u32()).collect::<Vec<_>>(),
            vec![
                1_608_637_542,
                3_421_126_067,
                4_083_286_876,
                787_846_414,
                3_143_890_026,
                3_348_747_335,
            ]
        );

        let mut doubles = Mt19937::new(42);
        for expected in [
            0.796_542_984_287_846,
            0.183_434_787_893_368_48,
            0.779_690_997_612_661_2,
        ] {
            assert_eq!(doubles.uniform_f64(), expected);
        }

        let mut floats = Mt19937::new(42);
        assert_eq!(
            (0..3).map(|_| floats.uniform_f32()).collect::<Vec<_>>(),
            vec![0.374_540_12, 0.796_543, 0.950_714_3]
        );
    }

    #[test]
    fn mt_distribution_matches_libcxx_candidate_selection() {
        let mut rng = Mt19937::new(42);
        assert_eq!(sample_distribution_mt(&[0.1, 0.2, 0.7], &mut rng), Ok(2));
        assert_eq!(sample_distribution_mt(&[0.1, 0.2, 0.7], &mut rng), Ok(1));
        assert_eq!(sample_distribution_mt(&[0.1, 0.2, 0.7], &mut rng), Ok(2));
    }

    #[test]
    fn mt_ordered_distribution_follows_ranked_candidates_not_token_ids() {
        let mut rng = Mt19937::new(42);
        assert_eq!(
            sample_distribution_mt_ordered(&[0.1, 0.2, 0.7], &[2, 1, 0], &mut rng),
            Ok(1)
        );
    }

    #[test]
    fn mt_snapshot_restores_the_exact_next_draw() {
        let mut original = Mt19937::new(0x1234_5678);
        for _ in 0..731 {
            original.next_u32();
        }
        let snapshot = original.snapshot();
        let expected = (0..32).map(|_| original.next_u32()).collect::<Vec<_>>();
        let mut restored = Mt19937::from_snapshot(&snapshot).unwrap();
        let actual = (0..32).map(|_| restored.next_u32()).collect::<Vec<_>>();
        assert_eq!(actual, expected);

        let mut corrupt = snapshot;
        corrupt.state.pop();
        assert_eq!(
            Mt19937::from_snapshot(&corrupt).unwrap_err(),
            SamplingError::InvalidRngState
        );
    }

    #[test]
    fn mt_speculative_decision_replays_from_snapshot() {
        let draft = vec![0, 1];
        let q = vec![vec![0.8, 0.2], vec![0.1, 0.9]];
        let p = vec![vec![0.4, 0.6], vec![0.7, 0.3], vec![0.25, 0.75]];
        let mut original = Mt19937::new(42);
        let snapshot = original.snapshot();
        let expected = verify_full_speculative_mt(&draft, &q, &p, &mut original).unwrap();
        let mut restored = Mt19937::from_snapshot(&snapshot).unwrap();
        let actual = verify_full_speculative_mt(&draft, &q, &p, &mut restored).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(restored.snapshot(), original.snapshot());
    }

    #[test]
    fn sparse_maximal_coupling_matches_dense_seed_for_seed() {
        fn sparse(probabilities: &[f32], order: &[u32]) -> SparseProbabilityRow {
            SparseProbabilityRow {
                vocab_size: probabilities.len().try_into().unwrap(),
                entries: order
                    .iter()
                    .copied()
                    .filter_map(|token| {
                        let probability = probabilities[token as usize];
                        (probability > 0.0).then_some(SparseProbabilityEntry { token, probability })
                    })
                    .collect(),
            }
        }

        let drafts = vec![0, 1];
        let q = vec![vec![0.8, 0.2, 0.0], vec![0.1, 0.9, 0.0]];
        let p = vec![
            vec![0.4, 0.6, 0.0],
            vec![0.7, 0.3, 0.0],
            vec![0.25, 0.75, 0.0],
        ];
        let target_orders = vec![vec![1, 0], vec![0, 1], vec![1, 0]];
        let draft_rows = vec![sparse(&q[0], &[0, 1]), sparse(&q[1], &[1, 0])];
        let target_rows = p
            .iter()
            .zip(&target_orders)
            .map(|(row, order)| sparse(row, order))
            .collect::<Vec<_>>();

        for seed in 0..1_000 {
            let mut dense_rng = Mt19937::new(seed);
            let dense =
                verify_full_speculative_mt_ordered(&drafts, &q, &p, &target_orders, &mut dense_rng)
                    .unwrap();
            let mut sparse_rng = Mt19937::new(seed);
            let sparse = verify_sparse_speculative_mt(
                &drafts,
                &draft_rows,
                &target_rows,
                40,
                &mut sparse_rng,
            )
            .unwrap();
            assert_eq!(sparse, dense, "seed {seed}");
            assert_eq!(sparse_rng.snapshot(), dense_rng.snapshot(), "seed {seed}");
        }
    }

    #[test]
    fn sparse_rows_reject_duplicate_or_nonfinite_support() {
        let duplicate = SparseProbabilityRow {
            vocab_size: 2,
            entries: vec![
                SparseProbabilityEntry {
                    token: 0,
                    probability: 0.5,
                },
                SparseProbabilityEntry {
                    token: 0,
                    probability: 0.5,
                },
            ],
        };
        assert_eq!(duplicate.validate(), Err(SamplingError::InvalidLogits));

        let nonfinite = SparseProbabilityRow {
            vocab_size: 2,
            entries: vec![SparseProbabilityEntry {
                token: 1,
                probability: f32::NAN,
            }],
        };
        assert_eq!(nonfinite.validate(), Err(SamplingError::InvalidLogits));

        let scaled = SparseProbabilityRow {
            vocab_size: 2,
            entries: vec![
                SparseProbabilityEntry {
                    token: 0,
                    probability: 0.25,
                },
                SparseProbabilityEntry {
                    token: 1,
                    probability: 0.25,
                },
            ],
        };
        assert_eq!(scaled.validate(), Err(SamplingError::InvalidLogits));

        let malformed = OrderedDistribution {
            probabilities: vec![1.0, 0.0],
            weights: vec![1.0, 0.0],
            order: vec![99],
        };
        assert_eq!(
            SparseProbabilityRow::from_ordered(&malformed),
            Err(SamplingError::Geometry)
        );

        let incomplete = OrderedDistribution {
            probabilities: vec![0.5, 0.5],
            weights: vec![0.5, 0.5],
            order: vec![0],
        };
        assert_eq!(
            SparseProbabilityRow::from_ordered(&incomplete),
            Err(SamplingError::Geometry)
        );
    }

    #[test]
    fn sparse_maximal_handles_disjoint_support_without_dense_q() {
        let q = SparseProbabilityRow {
            vocab_size: 1_000,
            entries: vec![SparseProbabilityEntry {
                token: 1,
                probability: 1.0,
            }],
        };
        let p = SparseProbabilityRow {
            vocab_size: 1_000,
            entries: vec![SparseProbabilityEntry {
                token: 999,
                probability: 1.0,
            }],
        };
        let bonus = SparseProbabilityRow {
            vocab_size: 1_000,
            entries: vec![SparseProbabilityEntry {
                token: 7,
                probability: 1.0,
            }],
        };
        assert_eq!(
            verify_sparse_speculative_mt(
                &[2],
                std::slice::from_ref(&q),
                &[p.clone(), bonus.clone()],
                1,
                &mut Mt19937::new(0),
            ),
            Err(SamplingError::Geometry)
        );
        for seed in 0..100 {
            let decision = verify_sparse_speculative_mt(
                &[1],
                std::slice::from_ref(&q),
                &[p.clone(), bonus.clone()],
                1,
                &mut Mt19937::new(seed),
            )
            .unwrap();
            assert_eq!(
                decision,
                SpeculativeDecision {
                    accepted: 0,
                    next_token: 999,
                }
            );
        }
        assert_eq!(q.validate_bounded(1), Ok(()));
        assert_eq!(q.validate_bounded(0), Err(SamplingError::Geometry));
    }

    #[test]
    fn full_residual_subtracts_every_draft_probability() {
        let drafts = vec![1];
        let q = vec![vec![0.7, 0.2, 0.1]];
        let p = vec![vec![0.1, 0.1, 0.8], vec![0.2, 0.3, 0.5]];
        let mut rng = StdRng::seed_from_u64(3);
        let decision = verify_full_speculative(&drafts, &q, &p, &mut rng).unwrap();
        assert_eq!(decision.accepted, 0);
        // max(p-q,0) has support only at token 2.
        assert_eq!(decision.next_token, 2);
    }

    #[test]
    fn canonical_distribution_applies_top_k_then_top_p() {
        let p = distribution(
            &[3.0, 2.0, 1.0, 0.0],
            SamplingParams {
                temperature: 1.0,
                top_p: 0.8,
                top_k: 3,
                typical_p: 1.0,
                min_p: 0.0,
                top_n_sigma: 0.0,
                min_keep: 0,
            },
        )
        .unwrap();
        assert!(p[0] > 0.0 && p[1] > 0.0);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_only_preserves_source_candidate_order_and_f32_division() {
        let logits = [0.125_000_01, 7.75, -1.125];
        let temperature = 0.73f32;
        let distribution = distribution_ordered(
            &logits,
            SamplingParams {
                temperature,
                top_p: 1.0,
                top_k: 0,
                typical_p: 1.0,
                min_p: 0.0,
                top_n_sigma: 0.0,
                min_keep: 0,
            },
        )
        .unwrap();
        assert_eq!(distribution.order, vec![0, 1, 2]);
        let scaled = logits.map(|logit| logit / temperature);
        let maximum = scaled.into_iter().fold(f32::NEG_INFINITY, f32::max);
        for (token, scaled_logit) in scaled.iter().copied().enumerate() {
            assert_eq!(
                distribution.weights[token].to_bits(),
                (scaled_logit - maximum).exp().to_bits()
            );
        }
    }

    #[test]
    fn zero_temperature_keeps_the_first_maximum_in_candidate_order() {
        let distribution = distribution_ordered(
            &[3.0, 1.0, 3.0],
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                typical_p: 1.0,
                min_p: 0.0,
                top_n_sigma: 0.0,
                min_keep: 0,
            },
        )
        .unwrap();
        assert_eq!(distribution.order, vec![0, 1, 2]);
        assert_eq!(distribution.weights, vec![1.0, 0.0, 0.0]);
        let mut rng = Mt19937::new(42);
        assert_eq!(
            sample_distribution_mt_ordered(&distribution.weights, &distribution.order, &mut rng),
            Ok(0)
        );
    }

    #[test]
    fn shared_gumbel_is_order_and_common_scale_invariant() {
        let seed = SharedGumbelSeed::from_u64(0xdec0_de01);
        for position in 0..64 {
            let baseline = sample_shared_gumbel_ordered(
                &[0.05, 0.15, 0.3, 0.5],
                &[0, 1, 2, 3],
                seed,
                position,
            )
            .unwrap();
            let reordered = sample_shared_gumbel_ordered(
                &[0.05, 0.15, 0.3, 0.5],
                &[3, 1, 0, 2],
                seed,
                position,
            )
            .unwrap();
            let rescaled =
                sample_shared_gumbel_ordered(&[0.5, 1.5, 3.0, 5.0], &[0, 1, 2, 3], seed, position)
                    .unwrap();
            assert_eq!(baseline, reordered);
            assert_eq!(baseline, rescaled);
        }
    }

    #[test]
    fn shared_gumbel_protocol_vector_and_reference_cap_are_stable() {
        let seed = SharedGumbelSeed::from_bytes([0x42; 32]);
        let lowest = open_unit_interval_from_u64(0);
        let highest = open_unit_interval_from_u64(u64::MAX);
        assert!(lowest > 0.0 && highest < 1.0 && lowest < highest);
        assert_eq!(
            shared_gumbel_uniform(seed, 7, 9).to_bits(),
            0x3fe1_a60e_a70f_7809
        );
        let too_wide = vec![1.0f32; MAX_SHARED_GUMBEL_REFERENCE_SUPPORT + 1];
        let order = (0..too_wide.len() as u32).collect::<Vec<_>>();
        assert_eq!(
            sample_shared_gumbel_ordered(&too_wide, &order, seed, 7),
            Err(SamplingError::InvalidLogits)
        );
    }

    #[test]
    fn shared_gumbel_has_the_requested_finite_sampler_marginal() {
        let weights = [0.1, 0.3, 0.6];
        let mut counts = [0usize; 3];
        let trials = 30_000usize;
        for seed in 0..trials as u64 {
            let token = sample_shared_gumbel_ordered(
                &weights,
                &[0, 1, 2],
                SharedGumbelSeed::from_u64(seed),
                7,
            )
            .unwrap();
            counts[token as usize] += 1;
        }
        for (observed, expected) in counts.into_iter().zip(weights) {
            let frequency = observed as f64 / trials as f64;
            assert!(
                (frequency - expected as f64).abs() < 0.015,
                "observed {frequency}, expected {expected}"
            );
        }
    }

    #[test]
    fn shared_gumbel_rejects_ambiguous_or_incomplete_support() {
        let seed = SharedGumbelSeed::from_u64(1);
        assert_eq!(
            sample_shared_gumbel_ordered(&[0.5, 0.5], &[0, 0], seed, 0),
            Err(SamplingError::InvalidLogits)
        );
        assert_eq!(
            sample_shared_gumbel_ordered(&[0.5, 0.5], &[0], seed, 0),
            Err(SamplingError::InvalidLogits)
        );
        assert_eq!(
            sample_shared_gumbel_ordered(&[0.0, 0.0], &[0, 1], seed, 0),
            Err(SamplingError::InvalidLogits)
        );
    }

    #[test]
    fn drafter_invariant_frontier_stops_at_mismatch_or_retains_bonus() {
        assert_eq!(
            verify_drafter_invariant_frontier(3, &[4, 8, 15], &[3, 4, 9, 99, 100]),
            Ok(DrafterInvariantFrontierDecision {
                accepted_drafts: 1,
                commit_input_count: 2,
                frontier_out: 9,
            })
        );
        assert_eq!(
            verify_drafter_invariant_frontier(3, &[4, 8, 15], &[3, 4, 8, 15, 16]),
            Ok(DrafterInvariantFrontierDecision {
                accepted_drafts: 3,
                commit_input_count: 4,
                frontier_out: 16,
            })
        );
        assert_eq!(
            verify_drafter_invariant_frontier(3, &[1], &[4, 1, 2]),
            Err(SamplingError::Geometry)
        );
        assert_eq!(
            verify_drafter_invariant_frontier(3, &[1], &[3, 1]),
            Err(SamplingError::Geometry)
        );
    }
}
