/-
IMO 1977 Problem 6: If f : ℕ → ℕ with f(n) > 0 for all n and f(f(n)) < f(n+1) for all n > 0,
then f(n) = n for all n > 0.
Source: miniF2F (https://github.com/openai/miniF2F)
-/
import Mathlib

theorem imo_1977_p6 (f : ℕ → ℕ) (h₀ : ∀ n, 0 < f n)
    (h₁ : ∀ n, 0 < n → f (f n) < f (n + 1)) :
    ∀ n, 0 < n → f n = n := by
  sorry
