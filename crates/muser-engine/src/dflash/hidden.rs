#[derive(Debug)]
pub struct DFlashHiddenCache {
    pub target_layer_ids: Vec<usize>,
    pub hidden_dim: usize,
    data: Vec<f32>,
    pub n_committed: usize,
    capture_base: usize,
    capture_tokens: usize,
}

impl DFlashHiddenCache {
    pub fn new(target_layer_ids: Vec<usize>, hidden_dim: usize) -> Self {
        assert!(!target_layer_ids.is_empty() && hidden_dim > 0);
        Self {
            target_layer_ids,
            hidden_dim,
            data: Vec::new(),
            n_committed: 0,
            capture_base: 0,
            capture_tokens: 0,
        }
    }
    pub fn token_width(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_dim
    }
    fn token_range(&self, token: usize) -> std::ops::Range<usize> {
        let start = token * self.token_width();
        start..start + self.token_width()
    }
    pub fn begin_capture(&mut self, tokens: usize) {
        self.capture_base = self.n_committed;
        self.capture_tokens = tokens;
        let needed = (self.capture_base + tokens) * self.token_width();
        self.data.resize(self.data.len().max(needed), 0.);
        let start = self.capture_base * self.token_width();
        self.data[start..needed].fill(0.);
    }
    pub fn write_rows(
        &mut self,
        absolute_layer: usize,
        token_pos: usize,
        rows: &[f32],
        tokens: usize,
    ) -> bool {
        let Some(layer_index) = self
            .target_layer_ids
            .iter()
            .position(|&l| l == absolute_layer)
        else {
            return false;
        };
        assert_eq!(rows.len(), tokens * self.hidden_dim);
        assert!(token_pos + tokens <= self.capture_tokens);
        for row in 0..tokens {
            let dst = (self.capture_base + token_pos + row) * self.token_width()
                + layer_index * self.hidden_dim;
            let src = row * self.hidden_dim;
            self.data[dst..dst + self.hidden_dim]
                .copy_from_slice(&rows[src..src + self.hidden_dim]);
        }
        true
    }
    pub fn write_token_major(&mut self, rows: &[f32], tokens: usize) {
        assert_eq!(rows.len(), tokens * self.token_width());
        assert!(tokens <= self.capture_tokens);
        let start = self.capture_base * self.token_width();
        self.data[start..start + rows.len()].copy_from_slice(rows);
    }
    pub fn commit(&mut self, tokens: usize) {
        assert!(tokens <= self.capture_tokens);
        self.n_committed = self.capture_base + tokens;
    }
    pub fn rollback_capture(&mut self) {
        self.n_committed = self.capture_base;
        self.capture_tokens = 0;
    }
    pub fn reset(&mut self) {
        self.n_committed = 0;
        self.capture_base = 0;
        self.capture_tokens = 0;
    }
    pub fn read_current_batch(&self) -> (Vec<f32>, usize) {
        let n = self.n_committed.saturating_sub(self.capture_base);
        if n == 0 {
            return (Vec::new(), 0);
        };
        let start = self.capture_base * self.token_width();
        (self.data[start..start + n * self.token_width()].to_vec(), n)
    }
    pub fn read_all(&self) -> Vec<f32> {
        self.data[..self.n_committed * self.token_width()].to_vec()
    }

    /// Mark an already-consumed committed prefix while retaining the newest
    /// rows as the next incremental batch.
    pub fn retain_current_suffix(&mut self, rows: usize) {
        assert!(rows <= self.n_committed);
        self.capture_base = self.n_committed - rows;
        self.capture_tokens = rows;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejected_rows_never_become_visible() {
        let mut c = DFlashHiddenCache::new(vec![1, 3], 2);
        c.begin_capture(2);
        c.write_rows(1, 0, &[1., 2., 3., 4.], 2);
        c.write_rows(3, 0, &[5., 6., 7., 8.], 2);
        c.commit(1);
        assert_eq!(c.read_all(), vec![1., 2., 5., 6.]);
        c.retain_current_suffix(1);
        assert_eq!(c.read_current_batch(), (vec![1., 2., 5., 6.], 1));
    }
}
