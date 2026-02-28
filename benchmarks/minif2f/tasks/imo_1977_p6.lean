import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1977_p6 (f : ℕ+ → ℕ+) (h₀ : ∀ n, f (f n) < f (n + 1)) :
  ∀ n, f n = n := by
  sorry
