import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_48
  (b : ℕ)
  (h₀ : 0 < b)
  (h₁ : Nat.digits b 57 = [1, 2, 3]) :
  b = 4 := by
  sorry
