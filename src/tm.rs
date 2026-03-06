use core::cmp::Ord;
use std::array;
use itertools::multizip;
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use bitvec::array::BitArray;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;

pub struct TMBuild<const I: usize>
{
    clauses: usize,
    threshold: i64,
    specificity: f64,
}

impl<const I: usize> TMBuild<I>
{
    pub fn new() -> Self {
        Self {
            clauses: 20,
            threshold: 10,
            specificity: 3.0,
        }
    }

    pub fn clauses(mut self, n: usize) -> Self {
        self.clauses = n;
        self
    }

    pub fn threshold(mut self, t: i64) -> Self {
        self.threshold = t;
        self
    }

    pub fn specificity(mut self, s: f64) -> Self {
        self.specificity = s;
        self
    }
}

impl<const I: usize> TMBuild<I>
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    type BDA = BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>;

    pub fn build(self) -> TMTrain<I> {
        assert!(self.clauses > 0, "clause count must be > 0");
        assert!(self.clauses % 2 == 0, "clause count must be even");
        assert!(self.threshold > 0, "threshold must be > 0");
        assert!(self.specificity > 1., "specificity must be > 1.0");

        let clauses = (0..self.clauses)
            .map(|_| Clause::<I>::new(400, self.specificity))
            .collect();

        TMTrain {
            clauses,
            threshold: self.threshold,
        }
    }
}

pub struct TMTrain<const I: usize>
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    clauses: Vec<Clause<I>>,
    threshold: i64,
}

impl<const I: usize> TMTrain<I>
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    pub type BDA= BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>;
    pub type BA= BitArray<[usize; (I + usize::BITS as usize - 1) / usize::BITS as usize]>;



    pub fn fit(&mut self, inputs: &mut [(Self::BA, bool)], epochs: usize) -> Vec<f64> {
        let mut accuracies = Vec::with_capacity(epochs);
        let mut rng: SmallRng = rand::make_rng();
        for _ in 0..epochs {
            inputs.shuffle(&mut rng);
            let correct = inputs.iter()
                .filter(|(input, target)| self.train(input, *target))
                .count();
            accuracies.push(correct as f64 / inputs.len() as f64);
        }
        accuracies
    }
    pub fn train(
        &mut self,
        _input: &Self::BA,
        target: bool,
    ) -> bool {
        let mut input: Self::BDA = BitArray::ZERO;
        let rev = !*_input;
        input[..I].copy_from_bitslice(&_input[..I]);
        input[I..(2 * I)].copy_from_bitslice(&rev[..I]);
        let clause_cutoff = self.clauses.len() / 2;
        let vote: i64 = self.clauses.par_iter().enumerate().map(|(ind, x)|{
            match (x.run(&input), ind < clause_cutoff) {
                (true , true) => 1,
                (true,  false) => -1,
                _=> 0,
            }
        }).sum::<i64>().clamp(-self.threshold, self.threshold);
        let correct = (vote >= 0) == target;
        let target_num = if target {self.threshold}else{-self.threshold};
        let distance = target_num.abs_diff(vote);
        let prob = distance as f64 / (2 * self.threshold) as f64;
        let uni: Uniform<f64> = Uniform::try_from(0. .. 1.).unwrap();
        let cutoff = self.clauses.len() / 2;
        self.clauses.iter_mut().enumerate().par_bridge().for_each(|(ind, x)| {
            let mut rng: SmallRng = rand::make_rng();
            let fb = uni.sample(&mut rng);
            if fb >= prob {return}
            if (ind < cutoff) == target  {
                x.type_one(&input);
            } else {
                x.type_two(&input);
            }
        });
        correct
    }
}

#[derive(Debug, Clone)]
pub struct TMInfer<const I: usize, const C: usize>
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    clauses: [ClauseInfer<I>; C],
    cutoff: usize
}

impl<const I: usize, const C: usize> From<TMTrain<I>> for TMInfer<I, C>
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    fn from(t: TMTrain<I>) -> Self {
        let mut iter = t.clauses.into_iter();
        let clauses: [ClauseInfer<I>; C] = std::array::from_fn(|_| iter.next().unwrap().save());
        Self {
            clauses,
            cutoff: C / 2,
        }
    }

}

impl<const I: usize, const C: usize> TMInfer<I, C> 
where
    [(); (I + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    pub type BA= BitArray<[usize; (I + usize::BITS as usize - 1) / usize::BITS as usize]>;
    pub type BDA= BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>;

    pub fn run(&self, _input: &Self::BA) -> bool {
        let mut input: Self::BDA = BitArray::ZERO;
        let rev = !*_input;
        input[..I].copy_from_bitslice(&_input[..I]);
        input[I..(2 * I)].copy_from_bitslice(&rev[..I]);
        self.clauses.par_iter().enumerate().map(|(ind, x)|{
            match (x.run(&input), ind < self.cutoff) {
                (true , true) => 1,
                (true,  false) => -1,
                _=> 0,
            }
        }).sum::<i64>() >= 0

    }
}

#[derive(Debug, Clone)]
struct ClauseInfer<const I: usize>
where
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    include_mask: BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    include_count: usize,
}

impl<const I: usize> ClauseInfer<I>
where
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    fn run(
        &self,
        expanded_input: &BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) -> bool {
        let matched = *expanded_input & self.include_mask;
        self.include_count - matched.count_ones() == 0
    }
}




/// Polarity is determined by position in the clause list: first half positive, second half negative.
#[derive(Debug, Clone)]
struct Clause<const I: usize>
where
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    automata: [Automaton; 2 * I],
    include_mask: BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    include_count: usize,
    strong_threshold: f64,
}

impl<const I: usize> Clause<I>
where
    [(); 2 * I]:,
    [(); (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    fn new(automaton_states: u16, s: f64) -> Self {
        Self {
            automata: array::from_fn(|_| Automaton::new(automaton_states)),
            include_mask: BitArray::ZERO,
            include_count: 0,
            strong_threshold: (s - 1.) / s,
        }
    }

    fn sync_include_mask(&mut self) {
        for (i, auto) in self.automata.iter().enumerate() {
            self.include_mask.set(i, auto.include());
        }
        self.include_count = self.include_mask.count_ones();
    }

    /// Returns true if all included literals are satisfied by the expanded input.
    fn run(
        &self,
        expanded_input: &BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) -> bool {
        let matched = *expanded_input & self.include_mask;
        self.include_count - matched.count_ones() == 0
    }

    fn save(self) -> ClauseInfer<I> {
        ClauseInfer { include_mask: self.include_mask, include_count: self.include_count }
    }

    fn type_one(
        &mut self,
        expanded_input: &BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) {
        let mut rng = rand::rng();
        let uni: Uniform<f64> = Uniform::try_from(0. .. 1.).unwrap();
        let matched = *expanded_input & self.include_mask;
        let clause_output = self.include_count - matched.count_ones() == 0;
        for (literal, automaton) in multizip((expanded_input, &mut self.automata)) {
            let r = uni.sample(&mut rng);
            let strong_feedback = r < self.strong_threshold;
            if clause_output && *literal.as_ref() {
                if strong_feedback {
                automaton.increment();
                }
            } 
            else {
                if !strong_feedback {
                    automaton.decrement();
                }
            }
        }
        self.sync_include_mask();
    }
    fn type_two(
        &mut self,
        expanded_input: &BitArray<[usize; (2 * I + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) {
        let matched = *expanded_input & self.include_mask;
        let clause_output = self.include_count - matched.count_ones() == 0;
        for (literal, included, automaton) in multizip((expanded_input, self.include_mask, &mut self.automata)) {
            if clause_output && !*literal.as_ref() && !included {
                automaton.increment();
            }
        }
        self.sync_include_mask();
    }

}

/// `STATES` must be even, >0, and <=256. Include threshold is at `STATES/2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Automaton {
    max: u16,
    include_limit: u16,
    state: u16,
}

impl Automaton {

    /// New automaton, starts in "exclude" (just below threshold).
    pub fn new(num_states: u16) -> Self {
        Self {
            max: num_states,
            include_limit: (num_states / 2),
            state: (num_states / 2) - 1,
        }
    }

    pub fn increment(&mut self) {
        self.state = self.max.min(self.state.saturating_add(1))
    }

    pub fn decrement(&mut self) {
        self.state = self.state.saturating_sub(1)
    }

    /// Returns `true` if in "include" action.
    pub fn include(&self) -> bool {
        self.state >= self.include_limit
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod automaton_tests {
    use super::*;

    #[test]
    fn initial_state_is_excluding() {
        let a = Automaton::new(256);
        assert_eq!(a.state, 127);
        assert!(!a.include());
    }

    #[test]
    fn increment_from_excluding() {
        let mut a = Automaton::new(256);
        a.increment();
        assert_eq!(a.state, 128);
        assert!(a.include());
    }

    #[test]
    fn increment_from_including() {
        let mut a = Automaton::new(256);
        a.state = 200;
        a.increment();
        assert_eq!(a.state, 201);
        assert!(a.include());
    }

    #[test]
    fn decrement_from_excluding() {
        let mut a = Automaton::new(256);
        a.decrement();
        assert_eq!(a.state, 126);
        assert!(!a.include());
    }

    #[test]
    fn decrement_from_including() {
        let mut a = Automaton::new(256);
        a.state = 200;
        a.decrement();
        assert_eq!(a.state, 199);
        assert!(a.include());
    }

    #[test]
    fn saturates_at_max() {
        let mut a = Automaton::new(256);
        a.state = 256;
        a.increment();
        assert_eq!(a.state, 256);
    }

    #[test]
    fn saturates_at_min() {
        let mut a = Automaton::new(256);
        a.state = 0;
        a.decrement();
        assert_eq!(a.state, 0);
    }

    #[test]
    fn increment_then_decrement_is_identity() {
        let mut a = Automaton::new(256);
        let before = a.state;
        a.increment();
        a.decrement();
        assert_eq!(a.state, before);
    }

    #[test]
    fn small_automaton() {
        let a = Automaton::new(8);
        assert_eq!(a.state, 3);
        assert!(!a.include());

        let mut a = Automaton::new(8);
        a.state = 4;
        assert!(a.include());
    }

    #[test]
    fn threshold_boundary() {
        let mut a = Automaton::new(256);
        a.state = 127;
        assert!(!a.include());
        a.state = 128;
        assert!(a.include());
    }

    #[test]
    fn decrement_at_include_boundary_flips_to_exclude() {
        let mut a = Automaton::new(256);
        a.state = 128; // just entered include
        assert!(a.include());
        a.decrement();
        assert_eq!(a.state, 127);
        assert!(!a.include());
    }

    #[test]
    fn full_state_traversal() {
        let mut a = Automaton::new(256);
        assert_eq!(a.state, 127);

        for _ in 0..127 { a.decrement(); }
        assert_eq!(a.state, 0);
        assert!(!a.include());

        a.decrement();
        assert_eq!(a.state, 0); // saturates at 0

        for _ in 0..128 { a.increment(); }
        assert_eq!(a.state, 128);
        assert!(a.include());

        for _ in 0..128 { a.increment(); }
        assert_eq!(a.state, 256);
        assert!(a.include());

        a.increment();
        assert_eq!(a.state, 256); // saturates at max
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn build_with_defaults() {
        let tm = TMBuild::<100>::new().build();
        assert_eq!(tm.clauses.len(), 20);
        assert_eq!(tm.threshold, 10);
    }

    #[test]
    fn build_with_custom_params() {
        let tm = TMBuild::<64>::new()
            .clauses(40)
            .threshold(15)
            .specificity(3.9)
            .build();
        assert_eq!(tm.clauses.len(), 40);
        assert_eq!(tm.threshold, 15);
    }

    #[test]
    fn clauses_split_half_positive_half_negative() {
        let tm = TMBuild::<10>::new().clauses(6).build();
        assert_eq!(tm.clauses.len(), 6);
        assert_eq!(tm.clauses.len() / 2, 3);
    }

    #[test]
    fn clause_has_correct_automata_count() {
        let tm = TMBuild::<10>::new().build();
        assert_eq!(tm.clauses[0].automata.len(), 20);
    }

    #[test]
    #[should_panic(expected = "clause count must be > 0")]
    fn panics_on_zero_clauses() {
        TMBuild::<10>::new().clauses(0).build();
    }

    #[test]
    #[should_panic(expected = "clause count must be even")]
    fn panics_on_odd_clauses() {
        TMBuild::<10>::new().clauses(3).build();
    }

    #[test]
    #[should_panic(expected = "threshold must be > 0")]
    fn panics_on_zero_threshold() {
        TMBuild::<10>::new().threshold(0).build();
    }

    #[test]
    #[should_panic(expected = "specificity must be > 1.0")]
    fn panics_on_zero_specificity() {
        TMBuild::<10>::new().specificity(0.0).build();
    }

    #[test]
    fn default_builder_values() {
        let builder = TMBuild::<10>::new();
        assert_eq!(builder.clauses, 20);
        assert_eq!(builder.threshold, 10);
        assert!((builder.specificity - 3.0).abs() < f64::EPSILON);
    }
}

#[cfg(test)]
mod clause_tests {
    use super::*;

    type TestClause = Clause<4>;
    type LitBits = BitArray<[usize; (2 * 4 + usize::BITS as usize - 1) / usize::BITS as usize]>;

    fn make_clause() -> TestClause {
        Clause::<4>::new(256, 3.0)
    }

    #[test]
    fn initial_mask_is_all_zeros() {
        let c = make_clause();
        assert_eq!(c.include_mask.count_ones(), 0);
        assert_eq!(c.include_count, 0);
    }

    #[test]
    fn sync_include_mask_reflects_automata() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.automata[3].state = 130;
        c.sync_include_mask();

        assert!(c.include_mask[0]);
        assert!(!c.include_mask[1]);
        assert!(!c.include_mask[2]);
        assert!(c.include_mask[3]);
        assert_eq!(c.include_mask.count_ones(), 2);
        assert_eq!(c.include_count, 2);
    }

    #[test]
    fn evaluate_all_included_pass() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.automata[2].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        input.set(2, true);

        assert_eq!(c.run(&input), true);
    }

    #[test]
    fn evaluate_some_failures() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.automata[1].state = 200;
        c.automata[2].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);

        assert_eq!(c.run(&input), false);
    }

    #[test]
    fn evaluate_empty_mask_is_zero_failures() {
        let c = make_clause();
        let input = LitBits::ZERO;
        assert_eq!(c.run(&input), true);
    }

    // Type I feedback uses internal randomness, so we can only test
    // that it doesn't panic and that the mask stays consistent.

    #[test]
    fn type_one_does_not_panic() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);

        c.type_one(&input);
    }

    #[test]
    fn type_one_keeps_mask_consistent() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.automata[2].state = 130;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        input.set(2, true);
        c.type_one(&input);

        for (i, auto) in c.automata.iter().enumerate() {
            assert_eq!(c.include_mask[i], auto.include(), "Mask mismatch at {}", i);
        }
        assert_eq!(c.include_count, c.include_mask.count_ones());
    }

    // Type II feedback

    #[test]
    fn type_two_fires_lit0_excluded_increments() {
        // clause fires (empty mask), lit=0, not included => increment: 127 → 128
        let mut c = make_clause();
        let input = LitBits::ZERO;

        c.type_two(&input);
        assert_eq!(c.automata[0].state, 128);
        assert!(c.automata[0].include());
    }

    #[test]
    fn type_two_fires_lit1_no_change() {
        // clause fires, lit=1 => no change
        let mut c = make_clause();
        let mut input = LitBits::ZERO;
        input.set(0, true);

        let before = c.automata[0].state;
        c.type_two(&input);
        assert_eq!(c.automata[0].state, before);
    }

    #[test]
    fn type_two_no_fire_no_change() {
        // clause doesn't fire => all automata unchanged
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let input = LitBits::ZERO;
        let states_before: Vec<u16> = c.automata.iter().map(|a| a.state).collect();
        c.type_two(&input);
        let states_after: Vec<u16> = c.automata.iter().map(|a| a.state).collect();
        assert_eq!(states_before, states_after);
    }

    #[test]
    fn run_ignores_non_included_bits() {
        let mut c = make_clause();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        input.set(3, true);
        input.set(5, true);
        input.set(7, true);
        assert!(c.run(&input));

        let mut minimal = LitBits::ZERO;
        minimal.set(0, true);
        assert_eq!(c.run(&input), c.run(&minimal));
    }
}

#[cfg(test)]
mod trainer_tests {
    use super::*;

    type InputBits = BitArray<[usize; (10 + usize::BITS as usize - 1) / usize::BITS as usize]>;

    #[test]
    fn train_does_not_panic() {
        let mut tm = TMBuild::<10>::new()
            .build();
        let input = InputBits::ZERO;
        tm.train(&input, true);
    }

    #[test]
    fn train_multiple_steps_no_panic() {
        let mut tm = TMBuild::<10>::new()
            .clauses(4)
            .build();
        let mut input = InputBits::ZERO;
        input.set(0, true);
        input.set(3, true);
        input.set(7, true);
        for i in 0..200 {
            tm.train(&input, i % 2 == 0);
        }
    }

    #[test]
    fn vote_clamp_stress() {
        let mut tm = TMBuild::<10>::new()
            .clauses(20)
            .threshold(5)
            .build();
        let input = InputBits::ZERO;
        for _ in 0..100 {
            tm.train(&input, true);
            tm.train(&input, false);
        }
    }

    #[test]
    fn train_returns_bool() {
        let mut tm = TMBuild::<10>::new()
            .build();
        let input = InputBits::ZERO;
        let result: bool = tm.train(&input, true);
        let _ = result;
    }

    #[test]
    fn fit_returns_per_epoch_accuracy() {
        let mut tm = TMBuild::<10>::new()
            .clauses(4)
            .build();
        let mut data = vec![
            (InputBits::ZERO, true),
            (InputBits::ZERO, false),
        ];
        let accuracies = tm.fit(&mut data, 5);
        assert_eq!(accuracies.len(), 5);
        for acc in &accuracies {
            assert!(*acc >= 0.0 && *acc <= 1.0);
        }
    }
}
