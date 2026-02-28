import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_3 : (∑ x ∈ Finset.range 9, (x + 1) ^ 2) % 10 = 5 := by
  sorry
