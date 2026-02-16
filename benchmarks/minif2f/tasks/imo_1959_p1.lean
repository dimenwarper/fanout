/-
IMO 1959 Problem 1: Prove that (21n+4) and (14n+3) are coprime for every natural number n.
Source: miniF2F (https://github.com/openai/miniF2F)
-/
import Mathlib

theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) :
    Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry
