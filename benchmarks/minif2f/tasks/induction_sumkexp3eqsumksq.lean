import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem induction_sumkexp3eqsumksq
  (n : ℕ) :
  (∑ k ∈ Finset.range n, k ^ 3) = (∑ k ∈ Finset.range n, k) ^ 2 := by
  sorry
