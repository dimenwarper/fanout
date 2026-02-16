/-
MATHD Algebra 478: Given v = (1/3)(b*h), b = 30, h = 13/2, prove v = 65.
Source: miniF2F (https://github.com/openai/miniF2F)
-/
import Mathlib

theorem mathd_algebra_478
    (b h v : ℝ)
    (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
    (h₁ : v = 1 / 3 * (b * h))
    (h₂ : b = 30)
    (h₃ : h = 13 / 2) :
    v = 65 := by
  sorry
