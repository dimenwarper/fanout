# miniF2F

Olympiad-level formal theorem proving benchmark adapted from [miniF2F](https://github.com/openai/miniF2F), using Lean 4 with Mathlib.

## Task

Given a Lean 4 theorem statement (with `sorry` as placeholder), produce a complete proof that typechecks.

## Tasks included

| Task | Source | Description |
|------|--------|-------------|
| `imo_1959_p1.lean` | IMO 1959 | gcd(21n+4, 14n+3) = 1 |
| `imo_1963_p5.lean` | IMO 1963 | cos(pi/7) - cos(2pi/7) + cos(3pi/7) = 1/2 |
| `imo_1977_p6.lean` | IMO 1977 | f(f(n)) < f(n+1) implies f = id |
| `imo_1990_p3.lean` | IMO 1990 | n^2 divides 2^n+1 implies n=3 |
| `mathd_algebra_478.lean` | MATHD | Basic algebra (warmup) |

## Prerequisites

Install [elan](https://github.com/leanprover/elan) (Lean version manager):

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

## Eval

```bash
chmod +x eval.sh

# Score a single proof
./eval.sh solution.lean

# With fanout RSA
fanout run "$(cat tasks/imo_1959_p1.lean)

Replace sorry with a valid Lean 4 proof. Output only the complete .lean file." \
  -m anthropic/claude-sonnet-4 -n 3 \
  -s rsa --k-agg 2 -r 3 \
  --eval-script "./eval.sh" --materializer file --file-ext .lean \
  -e script
```

## Scoring

Binary: 1.0 if the proof typechecks (no `sorry`, no errors), 0.0 otherwise.
