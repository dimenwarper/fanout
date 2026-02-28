import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_35 (p q : ℝ → ℝ) (h₀ : ∀ x, p x = 2 - x ^ 2)
  (h₁ : ∀ x : ℝ, q x = 6 / x) : p (q 2) = -7 := by
  sorry
