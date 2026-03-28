"""
Synthetic dataset generation + ground truth solver.

Generates propositional logic problems as typed lambda calculus problems.
The ground truth solver finds proofs (for evaluation only — the model never sees them).

No answer labels. The dataset is premises + query. That's it.
"""
from __future__ import annotations
import random
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

from tqdm import tqdm

from .ir import Type, Atom, Arrow, Pi, Prod, Sigma, Sum, Term, Var, App, Lam, Pair, Fst, Snd, Inl, Inr, Case, Bottom, Absurd, Neg, Context
from .type_checker import gate
from .tokenizer import encode_problem, term_to_tokens


@dataclass
class Problem:
    """A proof problem: premises (context) + goal type."""
    premises: List[Type]       # the assumptions (Γ)
    goal: Type                 # the proposition to prove
    # Ground truth (for evaluation only, not training)
    has_proof: bool            # does a proof exist?
    min_proof_length: int      # minimum proof term length in tokens (0 if no proof)
    ground_truth_proof: Optional[Term]  # a minimal proof (None if no proof)

    @property
    def context(self) -> Context:
        return tuple(self.premises)

    def to_input_tokens(self) -> List[int]:
        return encode_problem(self.premises, self.goal)


# ── Ground Truth Solver ──────────────────────────────────────────────

def solve(ctx: Context, goal: Type, max_depth: int = 10) -> Optional[Term]:
    """Find a proof term for the goal type in the given context.

    Uses iterative deepening depth-first search over the proof term space.
    Returns the shortest proof found, or None.
    """
    for depth in range(1, max_depth + 1):
        result = _search(ctx, goal, depth, set())
        if result is not None:
            return result
    return None


def _search(ctx: Context, goal: Type, depth: int, seen: set) -> Optional[Term]:
    """DFS proof search with depth limit."""
    if depth <= 0:
        return None

    key = (len(ctx), goal, depth)
    if key in seen:
        return None
    seen.add(key)

    # Strategy 1: Direct variable lookup
    for i, ty in enumerate(ctx):
        if ty == goal:
            return Var(i)

    # Strategy 2: Arrow introduction (Lam)
    if isinstance(goal, Pi):
        new_ctx = (goal.domain,) + ctx
        body = _search(new_ctx, goal.codomain, depth - 1, set())
        if body is not None:
            return Lam(body)

    # Strategy 3: Product introduction (Pair)
    if isinstance(goal, Sigma):
        fst = _search(ctx, goal.fst_type, depth - 1, set())
        if fst is not None:
            snd = _search(ctx, goal.snd_type, depth - 1, set())
            if snd is not None:
                return Pair(fst, snd)

    # Strategy 4: Sum introduction (Inl / Inr)
    if isinstance(goal, Sum):
        left = _search(ctx, goal.left, depth - 1, set())
        if left is not None:
            return Inl(left)
        right = _search(ctx, goal.right, depth - 1, set())
        if right is not None:
            return Inr(right)

    # Strategy 4b: Ex falso — if ⊥ in context, prove anything
    for i, ty in enumerate(ctx):
        if isinstance(ty, Bottom):
            return Absurd(Var(i))

    # Strategy 5: Direct application — arrow in context whose codomain matches
    for i, ty in enumerate(ctx):
        if isinstance(ty, Pi) and ty.codomain == goal:
            arg = _search(ctx, ty.domain, depth - 1, set())
            if arg is not None:
                return App(Var(i), arg)

    # Strategy 6: Chained application — arrow whose codomain is another arrow leading to goal
    for i, ty in enumerate(ctx):
        if isinstance(ty, Pi) and isinstance(ty.codomain, Pi) and ty.codomain.codomain == goal:
            arg1 = _search(ctx, ty.domain, depth - 1, set())
            if arg1 is not None:
                arg2 = _search(ctx, ty.codomain.domain, depth - 1, set())
                if arg2 is not None:
                    return App(App(Var(i), arg1), arg2)

    # Strategy 7: Product elimination (Fst / Snd)
    for i, ty in enumerate(ctx):
        if isinstance(ty, Sigma):
            if ty.fst_type == goal:
                return Fst(Var(i))
            if ty.snd_type == goal:
                return Snd(Var(i))
            # Deep elimination: Fst/Snd then apply
            if isinstance(ty.fst_type, Pi) and ty.fst_type.codomain == goal:
                arg = _search(ctx, ty.fst_type.domain, depth - 1, set())
                if arg is not None:
                    return App(Fst(Var(i)), arg)
            if isinstance(ty.snd_type, Pi) and ty.snd_type.codomain == goal:
                arg = _search(ctx, ty.snd_type.domain, depth - 1, set())
                if arg is not None:
                    return App(Snd(Var(i)), arg)

    # Strategy 8: Case analysis on sum types
    for i, ty in enumerate(ctx):
        if isinstance(ty, Sum):
            left_fn = _search(ctx, Arrow(ty.left, goal), depth - 1, set())
            if left_fn is not None:
                right_fn = _search(ctx, Arrow(ty.right, goal), depth - 1, set())
                if right_fn is not None:
                    return Case(Var(i), left_fn, right_fn)

    return None


# ── Problem Generation ───────────────────────────────────────────────

ATOM_NAMES = [chr(ord('A') + i) for i in range(10)]  # A-J


def _random_atom(rng: random.Random) -> Atom:
    return Atom(rng.choice(ATOM_NAMES))


def _random_type(rng: random.Random, max_depth: int = 2) -> Type:
    """Generate a random type of bounded depth."""
    if max_depth <= 0 or rng.random() < 0.5:
        return _random_atom(rng)
    kind = rng.randint(0, 2)
    if kind == 0:
        return Arrow(_random_type(rng, max_depth - 1), _random_type(rng, max_depth - 1))
    elif kind == 1:
        return Prod(_random_type(rng, max_depth - 1), _random_type(rng, max_depth - 1))
    else:
        return Sum(_random_type(rng, max_depth - 1), _random_type(rng, max_depth - 1))


def _make_problem(rng: random.Random, premises: List[Type], goal: Type) -> Problem:
    """Helper: solve and wrap into Problem."""
    rng.shuffle(premises)
    proof = solve(tuple(premises), goal)
    return Problem(
        premises=premises, goal=goal,
        has_proof=proof is not None,
        min_proof_length=len(term_to_tokens(proof)) if proof else 0,
        ground_truth_proof=proof,
    )


# ── Simple generators ────────────────────────────────────────────────

def generate_simple(rng: random.Random) -> Problem:
    """2-step modus ponens. A→B, A ⊢ B"""
    a, b = [Atom(x) for x in rng.sample(ATOM_NAMES, 2)]
    return _make_problem(rng, [Arrow(a, b), a], b)


def generate_medium(rng: random.Random) -> Problem:
    """3-5 step chain. A→B, B→C, ..., A ⊢ Z"""
    chain_len = rng.randint(3, 5)
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, min(chain_len + 1, len(ATOM_NAMES)))]
    premises = [atoms[0]]
    for i in range(len(atoms) - 1):
        premises.append(Arrow(atoms[i], atoms[i + 1]))
    return _make_problem(rng, premises, atoms[-1])


def generate_hard(rng: random.Random) -> Problem:
    """Longer chains with branching."""
    n_atoms = rng.randint(5, 8)
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, min(n_atoms, len(ATOM_NAMES)))]
    premises = [atoms[0]]
    for i in range(len(atoms) - 1):
        premises.append(Arrow(atoms[i], atoms[i + 1]))
    for _ in range(rng.randint(1, 3)):
        src, dst = rng.choice(atoms[:-1]), rng.choice(atoms[1:])
        if src != dst:
            premises.append(Arrow(src, dst))
    return _make_problem(rng, premises, atoms[-1])


def generate_trap(rng: random.Random) -> Problem:
    """Unprovable: variables appear but no path exists."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 5)]
    a, b, c, d, e = atoms
    premises = [Arrow(a, b), Arrow(b, c), a, Arrow(d, e)]
    return _make_problem(rng, premises, e)


# ── Conjunction generators ───────────────────────────────────────────

def generate_conjunction(rng: random.Random) -> Problem:
    """Basic ∧ problems."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 4)]
    a, b, c, d = atoms
    variant = rng.randint(0, 2)
    if variant == 0:
        return _make_problem(rng, [a, b], Prod(a, b))
    elif variant == 1:
        return _make_problem(rng, [Prod(a, b)], a)
    else:
        return _make_problem(rng, [Prod(a, b), Arrow(b, c)], c)


def generate_disjunction(rng: random.Random) -> Problem:
    """Basic ∨ problems."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 3)]
    a, b, c = atoms
    variant = rng.randint(0, 1)
    if variant == 0:
        return _make_problem(rng, [a], Sum(a, b))
    else:
        return _make_problem(rng, [Sum(a, b), Arrow(a, c), Arrow(b, c)], c)


# ── NEW: Advanced generators ─────────────────────────────────────────

def generate_long_chain(rng: random.Random) -> Problem:
    """6-9 step modus ponens chain."""
    chain_len = rng.randint(6, 9)
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, min(chain_len + 1, len(ATOM_NAMES)))]
    premises = [atoms[0]]
    for i in range(len(atoms) - 1):
        premises.append(Arrow(atoms[i], atoms[i + 1]))
    return _make_problem(rng, premises, atoms[-1])


def generate_nested_arrow(rng: random.Random) -> Problem:
    """Nested arrow types: (A→B)→C, higher-order implications."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 4)]
    a, b, c, d = atoms
    variant = rng.randint(0, 3)
    if variant == 0:
        # (A→B)→C, A→B ⊢ C
        return _make_problem(rng, [Arrow(Arrow(a, b), c), Arrow(a, b)], c)
    elif variant == 1:
        # A→(B→C), A, B ⊢ C
        return _make_problem(rng, [Arrow(a, Arrow(b, c)), a, b], c)
    elif variant == 2:
        # A→B→C, A ⊢ B→C  (partial application)
        return _make_problem(rng, [Arrow(a, Arrow(b, c)), a], Arrow(b, c))
    else:
        # ⊢ A → A  (identity, no premises)
        return _make_problem(rng, [], Arrow(a, a))


def generate_lambda_intro(rng: random.Random) -> Problem:
    """Problems requiring lambda introduction."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 4)]
    a, b, c, d = atoms
    variant = rng.randint(0, 4)
    if variant == 0:
        # ⊢ A → A (identity)
        return _make_problem(rng, [], Arrow(a, a))
    elif variant == 1:
        # ⊢ A → (B → A) (constant)
        return _make_problem(rng, [], Arrow(a, Arrow(b, a)))
    elif variant == 2:
        # B→C ⊢ A→(B→C)  (weaken)
        return _make_problem(rng, [Arrow(b, c)], Arrow(a, Arrow(b, c)))
    elif variant == 3:
        # A→B ⊢ A→B (trivial but needs Lam)
        return _make_problem(rng, [], Arrow(Arrow(a, b), Arrow(a, b)))
    else:
        # (A→B), (B→C) ⊢ A→C (composition)
        return _make_problem(rng, [Arrow(a, b), Arrow(b, c)], Arrow(a, c))


def generate_mixed_connectives(rng: random.Random) -> Problem:
    """Problems combining →, ∧, ∨ in one."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 5)]
    a, b, c, d, e = atoms
    variant = rng.randint(0, 5)
    if variant == 0:
        # (A∧B)→C, A, B ⊢ C
        return _make_problem(rng, [Arrow(Prod(a, b), c), a, b], c)
    elif variant == 1:
        # A→B, A→C ⊢ A→(B∧C)
        return _make_problem(rng, [Arrow(a, b), Arrow(a, c)], Arrow(a, Prod(b, c)))
    elif variant == 2:
        # (A∨B)→C ⊢ A→C
        return _make_problem(rng, [Arrow(Sum(a, b), c)], Arrow(a, c))
    elif variant == 3:
        # A→C, B→C ⊢ (A∨B)→C
        return _make_problem(rng, [Arrow(a, c), Arrow(b, c)], Arrow(Sum(a, b), c))
    elif variant == 4:
        # (A∧B), B→(C∨D), C→E, D→E ⊢ E
        return _make_problem(rng,
            [Prod(a, b), Arrow(b, Sum(c, d)), Arrow(c, e), Arrow(d, e)], e)
    else:
        # A, B→C ⊢ (A∧C) requires A, B→C, B — unprovable without B!
        return _make_problem(rng, [a, Arrow(b, c)], Prod(a, c))


def generate_deep_elimination(rng: random.Random) -> Problem:
    """Problems requiring Fst/Snd then further steps."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 5)]
    a, b, c, d, e = atoms
    variant = rng.randint(0, 3)
    if variant == 0:
        # (A∧(B→C)), B ⊢ C  (extract B→C from pair, apply)
        return _make_problem(rng, [Prod(a, Arrow(b, c)), b], c)
    elif variant == 1:
        # ((A→B)∧(B→C)), A ⊢ C  (extract both, chain)
        return _make_problem(rng, [Prod(Arrow(a, b), Arrow(b, c)), a], c)
    elif variant == 2:
        # (A∧B), (B∧C)→D ⊢ D requires proving B∧C... only B available
        # Let's do: (A∧B), (C∧D), (B→E) ⊢ E
        return _make_problem(rng, [Prod(a, b), Prod(c, d), Arrow(b, e)], e)
    else:
        # ((A∨B)∧(A→C)∧(B→C)) — everything in a product
        return _make_problem(rng, [Prod(Sum(a, b), Prod(Arrow(a, c), Arrow(b, c)))], c)


def generate_nested_case(rng: random.Random) -> Problem:
    """Nested sum types requiring deeper case analysis."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 5)]
    a, b, c, d, e = atoms
    variant = rng.randint(0, 2)
    if variant == 0:
        # (A∨B)∨C, A→D, B→D, C→D ⊢ D
        return _make_problem(rng,
            [Sum(Sum(a, b), c), Arrow(a, d), Arrow(b, d), Arrow(c, d)], d)
    elif variant == 1:
        # A∨(B∨C), A→D, B→D, C→D ⊢ D
        return _make_problem(rng,
            [Sum(a, Sum(b, c)), Arrow(a, d), Arrow(b, d), Arrow(c, d)], d)
    else:
        # (A∨B), (A→(C∨D)), (B→(C∨D)), C→E, D→E ⊢ E
        return _make_problem(rng,
            [Sum(a, b), Arrow(a, Sum(c, d)), Arrow(b, Sum(c, d)),
             Arrow(c, e), Arrow(d, e)], e)


def generate_complex_trap(rng: random.Random) -> Problem:
    """Complex unprovable problems — more deceptive than simple traps."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 6)]
    a, b, c, d, e, f = atoms
    variant = rng.randint(0, 3)
    if variant == 0:
        # A→B, C→D, A ⊢ D  (disconnect: A gives B, not D)
        return _make_problem(rng, [Arrow(a, b), Arrow(c, d), a], d)
    elif variant == 1:
        # (A∧B)→C, A ⊢ C  (need B too!)
        return _make_problem(rng, [Arrow(Prod(a, b), c), a], c)
    elif variant == 2:
        # A→B, B→C, D ⊢ C  (have D, need A to start chain)
        return _make_problem(rng, [Arrow(a, b), Arrow(b, c), d], c)
    else:
        # (A∨B)→C ⊢ A→C, plus A→D. Goal: D  (looks provable but isn't)
        return _make_problem(rng, [Arrow(Sum(a, b), c), Arrow(a, d)], d)


# ── Refutation generators — prove impossibility ──────────────────────

def generate_refutation_simple(rng: random.Random) -> Problem:
    """Prove ¬(A ∧ ¬A) — contradiction is impossible."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 2)]
    a, b = atoms
    variant = rng.randint(0, 3)
    if variant == 0:
        # ⊢ ¬(A ∧ ¬A) = (A ∧ (A→⊥)) → ⊥
        # proof: λp.(π₂(p) π₁(p))
        return _make_problem(rng, [], Neg(Prod(a, Neg(a))))
    elif variant == 1:
        # A, ¬A ⊢ ⊥
        # proof: (v1 v0)
        return _make_problem(rng, [a, Neg(a)], Bottom())
    elif variant == 2:
        # ¬A ⊢ ¬(A ∧ B) = (A ∧ B) → ⊥
        # proof: λp.(v0 π₁(p))
        return _make_problem(rng, [Neg(a)], Neg(Prod(a, b)))
    else:
        # A→B, ¬B ⊢ ¬A = A → ⊥
        # proof: λa.(v0 (v1 a))  — contrapositive
        return _make_problem(rng, [Arrow(a, b), Neg(b)], Neg(a))


def generate_refutation_medium(rng: random.Random) -> Problem:
    """Medium refutation: modus tollens, contradiction chains."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 3)]
    a, b, c = atoms
    variant = rng.randint(0, 3)
    if variant == 0:
        # Modus tollens: A→B, ¬B ⊢ ¬A
        return _make_problem(rng, [Arrow(a, b), Neg(b)], Neg(a))
    elif variant == 1:
        # Chain contrapositive: A→B, B→C, ¬C ⊢ ¬A
        return _make_problem(rng, [Arrow(a, b), Arrow(b, c), Neg(c)], Neg(a))
    elif variant == 2:
        # ¬(A ∨ B) ⊢ ¬A  (if neither, then not left)
        # ¬(A ∨ B) = (A ∨ B) → ⊥
        # proof: λa.(v0 (inl a))
        return _make_problem(rng, [Neg(Sum(a, b))], Neg(a))
    else:
        # ¬(A ∨ B) ⊢ ¬B
        return _make_problem(rng, [Neg(Sum(a, b))], Neg(b))


def generate_ex_falso(rng: random.Random) -> Problem:
    """From ⊥ prove anything — ex falso quodlibet."""
    atoms = [Atom(x) for x in rng.sample(ATOM_NAMES, 3)]
    a, b, c = atoms
    variant = rng.randint(0, 2)
    if variant == 0:
        # ⊥ ⊢ A
        return _make_problem(rng, [Bottom()], a)
    elif variant == 1:
        # ⊥ ⊢ A → B
        return _make_problem(rng, [Bottom()], Arrow(a, b))
    else:
        # A, ¬A ⊢ B  (contradiction gives anything)
        return _make_problem(rng, [a, Neg(a)], b)


# ── Dataset Generation ───────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "simple": 0.12,
    "medium": 0.10,
    "hard": 0.08,
    "trap": 0.03,
    "conjunction": 0.05,
    "disjunction": 0.05,
    "long_chain": 0.08,
    "nested_arrow": 0.06,
    "lambda_intro": 0.06,
    "mixed_connectives": 0.06,
    "deep_elimination": 0.04,
    "nested_case": 0.04,
    "complex_trap": 0.03,
    "refutation_simple": 0.08,
    "refutation_medium": 0.06,
    "ex_falso": 0.06,
}

ALL_GENERATORS = {
    "simple": generate_simple,
    "medium": generate_medium,
    "hard": generate_hard,
    "trap": generate_trap,
    "conjunction": generate_conjunction,
    "disjunction": generate_disjunction,
    "long_chain": generate_long_chain,
    "nested_arrow": generate_nested_arrow,
    "lambda_intro": generate_lambda_intro,
    "mixed_connectives": generate_mixed_connectives,
    "deep_elimination": generate_deep_elimination,
    "nested_case": generate_nested_case,
    "complex_trap": generate_complex_trap,
    "refutation_simple": generate_refutation_simple,
    "refutation_medium": generate_refutation_medium,
    "ex_falso": generate_ex_falso,
}


def generate_dataset(
    n_problems: int,
    seed: int = 42,
    difficulty_weights: Optional[Dict[str, float]] = None,
) -> List[Problem]:
    """Generate a dataset of proof problems."""
    if difficulty_weights is None:
        difficulty_weights = DEFAULT_WEIGHTS

    # Filter to only generators that exist
    generators = {k: ALL_GENERATORS[k] for k in difficulty_weights if k in ALL_GENERATORS}
    if not generators:
        generators = ALL_GENERATORS
        difficulty_weights = DEFAULT_WEIGHTS

    rng = random.Random(seed)
    difficulties = list(generators.keys())
    weights = [difficulty_weights.get(d, 0.0) for d in difficulties]

    problems = []
    for _ in tqdm(range(n_problems), desc="Generating problems", leave=False):
        difficulty = rng.choices(difficulties, weights=weights, k=1)[0]
        try:
            problem = generators[difficulty](rng)
            problems.append(problem)
        except (ValueError, IndexError):
            problem = generate_simple(rng)
            problems.append(problem)

    return problems


def dataset_stats(problems: List[Problem]) -> Dict:
    """Compute statistics about a dataset."""
    n_total = len(problems)
    n_provable = sum(1 for p in problems if p.has_proof)
    n_unprovable = n_total - n_provable
    proof_lengths = [p.min_proof_length for p in problems if p.has_proof and p.min_proof_length > 0]
    return {
        "total": n_total,
        "provable": n_provable,
        "unprovable": n_unprovable,
        "provable_pct": n_provable / n_total * 100 if n_total > 0 else 0,
        "avg_proof_length": sum(proof_lengths) / len(proof_lengths) if proof_lengths else 0,
        "max_proof_length": max(proof_lengths) if proof_lengths else 0,
        "min_proof_length": min(proof_lengths) if proof_lengths else 0,
    }
