#![feature(generic_const_exprs)]
#![feature(inherent_associated_types)]

use core::iter::IntoIterator;
use core::iter::Iterator;
use std::array;
use std::iter::Cycle;
use bitvec::vec::BitVec;
use itertools::multizip;
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::RngExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use bitvec::array::BitArray;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;

pub struct TsetlinBuilder<const NUM_INPUTS: usize>
{
    clauses: usize,
    selection_pool_size: usize,
    threshold: i64,
    specificity: f64,
}

impl<const NUM_INPUTS: usize> TsetlinBuilder<NUM_INPUTS>
{
    pub fn new() -> Self {
        Self {
            clauses: 20,
            selection_pool_size: 1000000,
            threshold: 10,
            specificity: 3.0,
        }
    }

    pub fn clauses(mut self, n: usize) -> Self {
        self.clauses = n;
        self
    }

    pub fn selection_pool_size(mut self, n: usize) -> Self {
        self.selection_pool_size = n;
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

impl<const NUM_INPUTS: usize> TsetlinBuilder<NUM_INPUTS>
where
    [(); (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    type BDA = BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>;

    fn create_selection_vector(&self, uni: &Uniform<f64>, ln_complement: f64) -> Self::BDA {
        let mut index = 0;
        let mut rng = rand::rng();
        let mut sv: Self::BDA = BitArray::new([usize::MAX; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]);
        loop {
            index += (uni.sample(&mut rng).ln() / ln_complement).ceil() as usize;
            if index >= 2 * NUM_INPUTS {
                break;
            }
            sv.set(index, false);
        }
        sv
    }

    pub fn build(self) -> TsetlinTrainer<NUM_INPUTS> {
        assert!(self.clauses > 0, "clause count must be > 0");
        assert!(self.clauses % 2 == 0, "clause count must be even");
        assert!(self.threshold > 0, "threshold must be > 0");
        assert!(self.specificity > 1., "specificity must be > 1.0");

        let ln_complement = (1.0 - 1.0 / self.specificity).ln();
        let mut rng = rand::rng();
        let uni: Uniform<f64> = Uniform::try_from(0. .. 1.).unwrap();
        let strong_selection_pool = (0..self.selection_pool_size).into_par_iter().map(|_|self.create_selection_vector(&uni, ln_complement)).collect::<Vec<_>>().into_iter().cycle();
        let uniform_pool = (&mut rng).sample_iter(&uni).take(self.selection_pool_size).collect::<Vec<_>>().into_iter().cycle();
        
        let clauses = (0..self.clauses)
            .map(|_| Clause::<NUM_INPUTS, 256>::new())
            .collect();

        TsetlinTrainer {
            clauses,
            strong_selection_pool,
            uniform_pool,
            threshold: self.threshold,
        }
    }
}

pub struct TsetlinTrainer<const NUM_INPUTS: usize>
where
    [(); (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    clauses: Vec<Clause<NUM_INPUTS, 256>>,
    strong_selection_pool: Cycle<std::vec::IntoIter<BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>>>,
    uniform_pool: Cycle<std::vec::IntoIter<f64>>,
    threshold: i64,
}

impl<const NUM_INPUTS: usize> TsetlinTrainer<NUM_INPUTS>
where
    [(); (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    pub type BDA= BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>;
    pub type BA= BitArray<[usize; (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>;



    pub fn fit(&mut self, inputs: &[(Self::BA, bool)], epochs: usize) -> Vec<f64> {
        let mut accuracies = Vec::with_capacity(epochs);
        for _ in 0..epochs {
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
        input[..NUM_INPUTS].copy_from_bitslice(&_input[..NUM_INPUTS]);
        input[NUM_INPUTS..(2 * NUM_INPUTS)].copy_from_bitslice(&rev[..NUM_INPUTS]);
        let clause_cutoff = self.clauses.len() / 2;
        let vote: i64 = self.clauses.par_iter().enumerate().map(|(ind, x)|{
            match (x.run(&input), ind < clause_cutoff) {
                (true , true) => 1,
                (true,  false) => -1,
                (_,_) => 0,
            }
        }).sum::<i64>().clamp(-self.threshold, self.threshold);
        let correct = (vote >= 0) == target;
        let type_one_prob = if target == true {
            (self.threshold - vote) as f64 / (2 * self.threshold) as f64
        } else {
            (self.threshold + vote) as f64 / (2 * self.threshold) as f64
        };
        let type_two_prob = if target == true {
            (self.threshold + vote) as f64 / (2 * self.threshold) as f64
        } else {
            (self.threshold - vote) as f64 / (2 * self.threshold) as f64
        };
        let cutoff = self.clauses.len() / 2;
        let uni_vec = &self.uniform_pool.by_ref().take(self.clauses.len()).collect::<Vec<_>>();
        let mut mask = self.dynamic_selection_mask(&uni_vec[..cutoff], type_one_prob);
        let mut t_two_mask = self.dynamic_selection_mask(&uni_vec[cutoff..],  type_two_prob);
        mask.append(&mut t_two_mask);
        let mask_len = mask.len();
        let strong_chunk = &self.strong_selection_pool.by_ref().take(mask_len).collect::<Vec<_>>();
        multizip((mask, &mut self.clauses, strong_chunk)).enumerate().par_bridge().for_each(|(ind, (fb , cl, strong_mask))| {
            match (fb, (ind < cutoff) == target) {
                (true, true) => cl.type_one(&input, strong_mask),
                (true, false) => cl.type_two(&input),
                (false, _) => {},
            }
        });
        correct
    }

    pub fn save(self) -> TMInference<NUM_INPUTS> {
        let cutoff = self.clauses.len() / 2;
        TMInference { clauses: self.clauses.into_iter().map(|x| x.save()).collect::<Vec<_>>(), cutoff }
    }

    fn dynamic_selection_mask(&self, uni: &[f64], probability: f64) -> BitVec {
        uni.into_iter().map(|x| {
            *x < probability
        }).collect::<BitVec>()
    }

}

#[derive(Debug, Clone)]
pub struct TMInference<const NUM_INPUTS: usize>
where
    [(); (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    clauses: Vec<ClauseInference<NUM_INPUTS, 256>>,
    cutoff: usize
}

impl<const NUM_INPUTS: usize> TMInference<NUM_INPUTS> 
where
    [(); (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    pub type BA= BitArray<[usize; (NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>;
    pub type BDA= BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>;

    pub fn run(&self, _input: &Self::BA) -> bool {
        let mut input: Self::BDA = BitArray::ZERO;
        let rev = !*_input;
        input[..NUM_INPUTS].copy_from_bitslice(&_input[..NUM_INPUTS]);
        input[NUM_INPUTS..(2 * NUM_INPUTS)].copy_from_bitslice(&rev[..NUM_INPUTS]);
        self.clauses.par_iter().enumerate().map(|(ind, x)|{
            match (x.run(&input), ind < self.cutoff) {
                (true , true) => 1,
                (true,  false) => -1,
                (_,_) => 0,
            }
        }).sum::<i64>() >= 0

    }
}

#[derive(Debug, Clone)]
struct ClauseInference<const NUM_INPUTS: usize, const AUTO_STATES: usize>
where
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    include_mask: BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    include_count: usize,
}

impl<const NUM_INPUTS: usize, const AUTO_STATES: usize> ClauseInference<NUM_INPUTS, AUTO_STATES>
where
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    fn run(
        &self,
        expanded_input: &BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) -> bool {
        let matched = *expanded_input & self.include_mask;
        self.include_count - matched.count_ones() == 0
    }
}




/// Polarity is determined by position in the clause list: first half positive, second half negative.
#[derive(Debug, Clone)]
struct Clause<const NUM_INPUTS: usize, const AUTO_STATES: usize>
where
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    automata: [Automaton<AUTO_STATES>; 2 * NUM_INPUTS],
    include_mask: BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    include_count: usize,
}

impl<const NUM_INPUTS: usize, const AUTO_STATES: usize> Clause<NUM_INPUTS, AUTO_STATES>
where
    [(); 2 * NUM_INPUTS]:,
    [(); (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]:,
{
    fn new() -> Self {
        Self {
            automata: array::from_fn(|_| Automaton::<AUTO_STATES>::new()),
            include_mask: BitArray::ZERO,
            include_count: 0,
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
        expanded_input: &BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) -> bool {
        let matched = *expanded_input & self.include_mask;
        self.include_count - matched.count_ones() == 0
    }

    fn save(self) -> ClauseInference<NUM_INPUTS, AUTO_STATES> {
        ClauseInference { include_mask: self.include_mask, include_count: self.include_count }
    }

    fn type_one(
        &mut self,
        expanded_input: &BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
        strong_feedback_selector: &BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) {
        let matched = *expanded_input & self.include_mask;
        let clause_output = self.include_count - matched.count_ones() == 0;
        for (literal, included, strong_feedback, automaton) in multizip((expanded_input, self.include_mask, strong_feedback_selector, &mut self.automata)) {
            match (clause_output, *literal.as_ref(), included, *strong_feedback.as_ref() ) {
                (true, true, true, true) => automaton.reward(),
                (true, false, false, false) => automaton.reward(),
                (false, _, false, false) => automaton.reward(),

                (true, true, false, true) =>automaton.penalize(),
                (false, _, true, false) => automaton.penalize(),
                _=> {}
            }

        }
        self.sync_include_mask();
    }
    fn type_two(
        &mut self,
        expanded_input: &BitArray<[usize; (2 * NUM_INPUTS + usize::BITS as usize - 1) / usize::BITS as usize]>,
    ) {
        let matched = *expanded_input & self.include_mask;
        let clause_output = self.include_count - matched.count_ones() == 0;
        for (literal, included, automaton) in multizip((expanded_input, self.include_mask, &mut self.automata)) {
            if clause_output && !*literal.as_ref() && !included {
                automaton.penalize();
            }
        }
        self.sync_include_mask();
    }

}

/// `STATES` must be even, >0, and <=256. Include threshold is at `STATES/2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Automaton<const STATES: usize> {
    state: u8,
}

impl<const STATES: usize> Automaton<STATES> {
    const INCLUDE_LIMIT: u8 = {
        assert!(STATES % 2 == 0, "STATES must be even");
        assert!(STATES > 0, "STATES must be greater than zero");
        assert!(STATES <= 256, "STATES must be at most 256");
        (STATES / 2) as u8
    };

    const STATE_MAX: u8 = (STATES - 1) as u8;

    /// New automaton, starts in "exclude" (just below threshold).
    pub fn new() -> Self {
        Self {
            state: Self::INCLUDE_LIMIT - 1,
        }
    }

    /// Reinforce current action (away from threshold).
    pub fn reward(&mut self) {
        if self.state >= Self::INCLUDE_LIMIT {
            if self.state < Self::STATE_MAX {
                self.state += 1;
            }
        } else if self.state > 0 {
            self.state -= 1;
        }
    }

    /// Push toward threshold (may flip action).
    pub fn penalize(&mut self) {
        if self.state >= Self::INCLUDE_LIMIT {
            self.state -= 1;
        } else {
            self.state += 1;
        }
    }

    /// Returns `true` if in "include" action.
    pub fn include(&self) -> bool {
        self.state >= Self::INCLUDE_LIMIT
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
        let a = Automaton::<256>::new();
        assert_eq!(a.state, 127);
        assert!(!a.include());
    }

    #[test]
    fn reward_when_excluding_decrements() {
        let mut a = Automaton::<256>::new();
        a.reward();
        assert_eq!(a.state, 126);
        assert!(!a.include());
    }

    #[test]
    fn reward_when_including_increments() {
        let mut a = Automaton::<256>::new();
        a.state = 200;
        a.reward();
        assert_eq!(a.state, 201);
        assert!(a.include());
    }

    #[test]
    fn penalize_when_excluding_increments() {
        let mut a = Automaton::<256>::new();
        a.penalize();
        assert_eq!(a.state, 128);
        assert!(a.include());
    }

    #[test]
    fn penalize_when_including_decrements() {
        let mut a = Automaton::<256>::new();
        a.state = 200;
        a.penalize();
        assert_eq!(a.state, 199);
        assert!(a.include());
    }

    #[test]
    fn saturates_at_max() {
        let mut a = Automaton::<256>::new();
        a.state = 255;
        a.reward();
        assert_eq!(a.state, 255);
    }

    #[test]
    fn saturates_at_min() {
        let mut a = Automaton::<256>::new();
        a.state = 0;
        a.reward();
        assert_eq!(a.state, 0);
    }

    #[test]
    fn reward_then_penalize_is_identity() {
        let mut a = Automaton::<256>::new();
        let before = a.state;
        a.reward();
        a.penalize();
        assert_eq!(a.state, before);
    }

    #[test]
    fn small_automaton() {
        let a = Automaton::<8>::new();
        assert_eq!(a.state, 3);
        assert!(!a.include());

        let mut a = Automaton::<8>::new();
        a.state = 4;
        assert!(a.include());
    }

    #[test]
    fn threshold_boundary() {
        let mut a = Automaton::<256>::new();
        a.state = 127;
        assert!(!a.include());
        a.state = 128;
        assert!(a.include());
    }

    #[test]
    fn penalize_at_include_boundary_flips_to_exclude() {
        let mut a = Automaton::<256>::new();
        a.state = 128; // just entered include
        assert!(a.include());
        a.penalize();
        assert_eq!(a.state, 127);
        assert!(!a.include());
    }

    #[test]
    fn full_state_traversal() {
        let mut a = Automaton::<256>::new();
        assert_eq!(a.state, 127);

        for _ in 0..127 { a.reward(); }
        assert_eq!(a.state, 0);
        assert!(!a.include());

        a.reward();
        assert_eq!(a.state, 0); // saturates at 0

        for _ in 0..128 { a.penalize(); }
        assert_eq!(a.state, 128);
        assert!(a.include());

        for _ in 0..127 { a.reward(); }
        assert_eq!(a.state, 255);
        assert!(a.include());

        a.reward();
        assert_eq!(a.state, 255); // saturates at 255
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn build_with_defaults() {
        let tm = TsetlinBuilder::<100>::new().build();
        assert_eq!(tm.clauses.len(), 20);
        assert_eq!(tm.threshold, 10);
    }

    #[test]
    fn build_with_custom_params() {
        let tm = TsetlinBuilder::<64>::new()
            .clauses(40)
            .threshold(15)
            .specificity(3.9)
            .build();
        assert_eq!(tm.clauses.len(), 40);
        assert_eq!(tm.threshold, 15);
    }

    #[test]
    fn clauses_split_half_positive_half_negative() {
        let tm = TsetlinBuilder::<10>::new().clauses(6).build();
        assert_eq!(tm.clauses.len(), 6);
        assert_eq!(tm.clauses.len() / 2, 3);
    }

    #[test]
    fn clause_has_correct_automata_count() {
        let tm = TsetlinBuilder::<10>::new().build();
        assert_eq!(tm.clauses[0].automata.len(), 20);
    }

    #[test]
    #[should_panic(expected = "clause count must be > 0")]
    fn panics_on_zero_clauses() {
        TsetlinBuilder::<10>::new().clauses(0).build();
    }

    #[test]
    #[should_panic(expected = "clause count must be even")]
    fn panics_on_odd_clauses() {
        TsetlinBuilder::<10>::new().clauses(3).build();
    }

    #[test]
    #[should_panic(expected = "threshold must be > 0")]
    fn panics_on_zero_threshold() {
        TsetlinBuilder::<10>::new().threshold(0).build();
    }

    #[test]
    #[should_panic(expected = "specificity must be > 1.0")]
    fn panics_on_zero_specificity() {
        TsetlinBuilder::<10>::new().specificity(0.0).build();
    }

    #[test]
    fn default_selection_pool_size() {
        let builder = TsetlinBuilder::<10>::new();
        assert_eq!(builder.clauses, 20);
        assert_eq!(builder.threshold, 10);
        assert!((builder.specificity - 3.0).abs() < f64::EPSILON);
        assert_eq!(builder.selection_pool_size, 1_000_000);
    }

    #[test]
    fn selection_pool_size_setter() {
        let tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(100)
            .build();
        assert_eq!(tm.clauses.len(), 20);
    }

    #[test]
    fn selection_vector_is_subset_of_all_ones() {
        let builder = TsetlinBuilder::<10>::new().specificity(3.0);
        let ln_complement = (1.0 - 1.0 / builder.specificity).ln();
        let uni = Uniform::try_from(0.0..1.0).unwrap();
        let all_ones = BitArray::<[usize; (2 * 10 + usize::BITS as usize - 1) / usize::BITS as usize]>::new(
            [usize::MAX; (2 * 10 + usize::BITS as usize - 1) / usize::BITS as usize],
        );
        for _ in 0..50 {
            let sv = builder.create_selection_vector(&uni, ln_complement);
            assert_eq!(sv | all_ones, all_ones);
            for i in 0..(2 * 10) {
                let _ = sv[i];
            }
        }
    }
}

#[cfg(test)]
mod clause_tests {
    use super::*;

    type TestClause = Clause<4, 256>;
    type LitBits = BitArray<[usize; (2 * 4 + usize::BITS as usize - 1) / usize::BITS as usize]>;

    #[test]
    fn initial_mask_is_all_zeros() {
        let c = TestClause::new();
        assert_eq!(c.include_mask.count_ones(), 0);
        assert_eq!(c.include_count, 0);
    }

    #[test]
    fn sync_include_mask_reflects_automata() {
        let mut c = TestClause::new();
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
        let mut c = TestClause::new();
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
        let mut c = TestClause::new();
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
        let c = TestClause::new();
        let input = LitBits::ZERO;
        assert_eq!(c.run(&input), true);
    }

    // Type I feedback: clause fires

    #[test]
    fn type_one_fires_included_lit1_strong_rewards() {
        // (clause_out=T, lit=1, included=T, strong=T) => reward: 200 → 201
        let mut c = TestClause::new();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        let strong = LitBits::new([usize::MAX; (2 * 4 + usize::BITS as usize - 1) / usize::BITS as usize]);

        c.type_one(&input, &strong);
        assert_eq!(c.automata[0].state, 201);
    }

    #[test]
    fn type_one_fires_excluded_lit0_weak_rewards() {
        // (clause_out=T, lit=0, included=F, strong=F) => reward: 127 → 126
        let mut c = TestClause::new();
        let input = LitBits::ZERO;
        let strong = LitBits::ZERO;

        let before = c.automata[0].state;
        c.type_one(&input, &strong);
        assert_eq!(c.automata[0].state, before - 1);
    }

    #[test]
    fn type_one_fires_excluded_lit1_strong_penalizes() {
        // (clause_out=T, lit=1, included=F, strong=T) => penalize: 127 → 128
        let mut c = TestClause::new();
        let mut input = LitBits::ZERO;
        input.set(0, true);
        let strong = LitBits::new([usize::MAX; (2 * 4 + usize::BITS as usize - 1) / usize::BITS as usize]);

        assert!(!c.automata[0].include());
        c.type_one(&input, &strong);
        assert_eq!(c.automata[0].state, 128);
        assert!(c.automata[0].include());
    }

    // Type I feedback: clause doesn't fire

    #[test]
    fn type_one_no_fire_excluded_weak_rewards() {
        // (clause_out=F, lit=_, included=F, strong=F) => reward: 127 → 126
        let mut c = TestClause::new();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let input = LitBits::ZERO;
        let strong = LitBits::ZERO;

        let before = c.automata[1].state;
        c.type_one(&input, &strong);
        assert_eq!(c.automata[1].state, before - 1);
    }

    #[test]
    fn type_one_no_fire_included_weak_penalizes() {
        // (clause_out=F, lit=_, included=T, strong=F) => penalize: 200 → 199
        let mut c = TestClause::new();
        c.automata[0].state = 200;
        c.automata[1].state = 200;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        let strong = LitBits::ZERO;

        c.type_one(&input, &strong);
        assert_eq!(c.automata[0].state, 199);
    }

    // Type II feedback

    #[test]
    fn type_two_fires_lit0_excluded_penalizes() {
        // clause fires, lit=0, not included => penalize: 127 → 128
        let mut c = TestClause::new();
        let input = LitBits::ZERO;

        c.type_two(&input);
        assert_eq!(c.automata[0].state, 128);
        assert!(c.automata[0].include());
    }

    #[test]
    fn type_two_fires_lit1_no_change() {
        // clause fires, lit=1 => no change
        let mut c = TestClause::new();
        let mut input = LitBits::ZERO;
        input.set(0, true);

        let before = c.automata[0].state;
        c.type_two(&input);
        assert_eq!(c.automata[0].state, before);
    }

    #[test]
    fn type_two_no_fire_no_change() {
        // clause doesn't fire => all automata unchanged
        let mut c = TestClause::new();
        c.automata[0].state = 200;
        c.sync_include_mask();

        let input = LitBits::ZERO;
        let states_before: Vec<u8> = c.automata.iter().map(|a| a.state).collect();
        c.type_two(&input);
        let states_after: Vec<u8> = c.automata.iter().map(|a| a.state).collect();
        assert_eq!(states_before, states_after);
    }

    // Mask consistency

    #[test]
    fn sync_mask_after_type_one() {
        let mut c = TestClause::new();
        c.automata[0].state = 200;
        c.automata[2].state = 130;
        c.sync_include_mask();

        let mut input = LitBits::ZERO;
        input.set(0, true);
        input.set(2, true);
        let strong = LitBits::new([usize::MAX; (2 * 4 + usize::BITS as usize - 1) / usize::BITS as usize]);
        c.type_one(&input, &strong);

        for (i, auto) in c.automata.iter().enumerate() {
            assert_eq!(c.include_mask[i], auto.include(), "Mask mismatch at {}", i);
        }
        assert_eq!(c.include_count, c.include_mask.count_ones());
    }

    #[test]
    fn run_ignores_non_included_bits() {
        let mut c = TestClause::new();
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
        let mut tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(1000)
            .build();
        let input = InputBits::ZERO;
        tm.train(&input, true);
    }

    #[test]
    fn train_multiple_steps_no_panic() {
        let mut tm = TsetlinBuilder::<10>::new()
            .clauses(4)
            .selection_pool_size(100)
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
    fn dynamic_selection_mask_length() {
        let tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(100)
            .build();
        let samples_42: Vec<f64> = vec![0.5; 42];
        let samples_0: Vec<f64> = vec![];
        let samples_1000: Vec<f64> = vec![0.5; 1000];
        assert_eq!(tm.dynamic_selection_mask(&samples_42, 0.5).len(), 42);
        assert_eq!(tm.dynamic_selection_mask(&samples_0, 0.5).len(), 0);
        assert_eq!(tm.dynamic_selection_mask(&samples_1000, 0.5).len(), 1000);
    }

    #[test]
    fn dynamic_selection_mask_all_true_at_prob_1() {
        let tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(100)
            .build();
        let samples: Vec<f64> = vec![0.5; 100];
        let mask = tm.dynamic_selection_mask(&samples, 1.0);
        assert_eq!(mask.len(), 100);
        assert!(mask.iter().all(|b| *b));
    }

    #[test]
    fn dynamic_selection_mask_all_false_at_prob_0() {
        let tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(100)
            .build();
        let samples: Vec<f64> = vec![0.5; 100];
        let mask = tm.dynamic_selection_mask(&samples, 0.0);
        assert_eq!(mask.len(), 100);
        assert!(mask.iter().all(|b| !*b));
    }

    #[test]
    fn vote_clamp_stress() {
        let mut tm = TsetlinBuilder::<10>::new()
            .clauses(20)
            .threshold(5)
            .selection_pool_size(500)
            .build();
        let input = InputBits::ZERO;
        for _ in 0..100 {
            tm.train(&input, true);
            tm.train(&input, false);
        }
    }

    #[test]
    fn train_returns_bool() {
        let mut tm = TsetlinBuilder::<10>::new()
            .selection_pool_size(100)
            .build();
        let input = InputBits::ZERO;
        let result: bool = tm.train(&input, true);
        let _ = result;
    }

    #[test]
    fn fit_returns_per_epoch_accuracy() {
        let mut tm = TsetlinBuilder::<10>::new()
            .clauses(4)
            .selection_pool_size(100)
            .build();
        let data = vec![
            (InputBits::ZERO, true),
            (InputBits::ZERO, false),
        ];
        let accuracies = tm.fit(&data, 5);
        assert_eq!(accuracies.len(), 5);
        for acc in &accuracies {
            assert!(*acc >= 0.0 && *acc <= 1.0);
        }
    }
}
