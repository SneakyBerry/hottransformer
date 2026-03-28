"""
Unified Type Checker — THE GATE.

Handles both STLC (propositional logic) and dependent types.
Pi subsumes Arrow, Sigma subsumes Prod.

Under Curry-Howard, type checking IS proof verification:
  check(Γ, t, A) = True  ⟺  t is a valid proof of A from assumptions Γ
"""
from __future__ import annotations
from dataclasses import dataclass
from .ir import (
    Term, Var, App, Lam,
    Pi, Sigma, Pair, Fst, Snd,
    Sum, Inl, Inr, Case,
    Id, Refl, J,
    Nat, Zero, Suc, NatElim,
    Bottom, Absurd,
    Universe, Set, Prop, Atom,
    Context, context_lookup, context_extend,
)


class TypeCheckError(Exception):
    pass


# ── Substitution (for dependent types) ──────────────────────────────

def subst(term: Term, index: int, replacement: Term) -> Term:
    """Substitute replacement for Var(index) in term, shift others."""
    if isinstance(term, Var):
        if term.index == index:
            return replacement
        elif term.index > index:
            return Var(term.index - 1)
        return term
    elif isinstance(term, App):
        return App(subst(term.func, index, replacement), subst(term.arg, index, replacement))
    elif isinstance(term, Lam):
        return Lam(subst(term.body, index + 1, _shift(replacement, 0, 1)))
    elif isinstance(term, Pi):
        return Pi(subst(term.domain, index, replacement),
                  subst(term.codomain, index + 1, _shift(replacement, 0, 1)))
    elif isinstance(term, Sigma):
        return Sigma(subst(term.fst_type, index, replacement),
                     subst(term.snd_type, index + 1, _shift(replacement, 0, 1)))
    elif isinstance(term, Pair):
        return Pair(subst(term.fst, index, replacement), subst(term.snd, index, replacement))
    elif isinstance(term, Fst):
        return Fst(subst(term.pair, index, replacement))
    elif isinstance(term, Snd):
        return Snd(subst(term.pair, index, replacement))
    elif isinstance(term, Sum):
        return Sum(subst(term.left, index, replacement), subst(term.right, index, replacement))
    elif isinstance(term, Inl):
        return Inl(subst(term.term, index, replacement))
    elif isinstance(term, Inr):
        return Inr(subst(term.term, index, replacement))
    elif isinstance(term, Case):
        return Case(subst(term.scrutinee, index, replacement),
                    subst(term.left_branch, index, replacement),
                    subst(term.right_branch, index, replacement))
    elif isinstance(term, Id):
        return Id(subst(term.ty, index, replacement),
                  subst(term.lhs, index, replacement),
                  subst(term.rhs, index, replacement))
    elif isinstance(term, Suc):
        return Suc(subst(term.n, index, replacement))
    return term


def _shift(term: Term, cutoff: int, amount: int) -> Term:
    if isinstance(term, Var):
        return Var(term.index + amount) if term.index >= cutoff else term
    elif isinstance(term, App):
        return App(_shift(term.func, cutoff, amount), _shift(term.arg, cutoff, amount))
    elif isinstance(term, Lam):
        return Lam(_shift(term.body, cutoff + 1, amount))
    elif isinstance(term, Pi):
        return Pi(_shift(term.domain, cutoff, amount), _shift(term.codomain, cutoff + 1, amount))
    elif isinstance(term, Sigma):
        return Sigma(_shift(term.fst_type, cutoff, amount), _shift(term.snd_type, cutoff + 1, amount))
    elif isinstance(term, Pair):
        return Pair(_shift(term.fst, cutoff, amount), _shift(term.snd, cutoff, amount))
    elif isinstance(term, Fst):
        return Fst(_shift(term.pair, cutoff, amount))
    elif isinstance(term, Snd):
        return Snd(_shift(term.pair, cutoff, amount))
    elif isinstance(term, Sum):
        return Sum(_shift(term.left, cutoff, amount), _shift(term.right, cutoff, amount))
    elif isinstance(term, Inl):
        return Inl(_shift(term.term, cutoff, amount))
    elif isinstance(term, Inr):
        return Inr(_shift(term.term, cutoff, amount))
    elif isinstance(term, Case):
        return Case(_shift(term.scrutinee, cutoff, amount),
                    _shift(term.left_branch, cutoff, amount),
                    _shift(term.right_branch, cutoff, amount))
    elif isinstance(term, Id):
        return Id(_shift(term.ty, cutoff, amount),
                  _shift(term.lhs, cutoff, amount),
                  _shift(term.rhs, cutoff, amount))
    elif isinstance(term, Suc):
        return Suc(_shift(term.n, cutoff, amount))
    return term


def subst_top(body: Term, arg: Term) -> Term:
    return subst(body, 0, arg)


# ── Normalization ────────────────────────────────────────────────────

def normalize(term: Term, fuel: int = 500) -> Term:
    if fuel <= 0:
        return term
    if isinstance(term, App):
        func = normalize(term.func, fuel - 1)
        arg = normalize(term.arg, fuel - 1)
        if isinstance(func, Lam):
            return normalize(subst_top(func.body, arg), fuel - 1)
        return App(func, arg)
    if isinstance(term, Fst):
        p = normalize(term.pair, fuel - 1)
        if isinstance(p, Pair):
            return normalize(p.fst, fuel - 1)
        return Fst(p)
    if isinstance(term, Snd):
        p = normalize(term.pair, fuel - 1)
        if isinstance(p, Pair):
            return normalize(p.snd, fuel - 1)
        return Snd(p)
    if isinstance(term, Lam):
        return Lam(normalize(term.body, fuel - 1))
    if isinstance(term, Pi):
        return Pi(normalize(term.domain, fuel - 1), normalize(term.codomain, fuel - 1))
    if isinstance(term, Sigma):
        return Sigma(normalize(term.fst_type, fuel - 1), normalize(term.snd_type, fuel - 1))
    if isinstance(term, Pair):
        return Pair(normalize(term.fst, fuel - 1), normalize(term.snd, fuel - 1))
    if isinstance(term, Id):
        return Id(normalize(term.ty, fuel - 1), normalize(term.lhs, fuel - 1), normalize(term.rhs, fuel - 1))
    if isinstance(term, Suc):
        return Suc(normalize(term.n, fuel - 1))
    return term


def terms_equal(a: Term, b: Term) -> bool:
    a, b = normalize(a), normalize(b)
    return _eq(a, b)


def _eq(a: Term, b: Term) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, Var): return a.index == b.index
    if isinstance(a, App): return _eq(a.func, b.func) and _eq(a.arg, b.arg)
    if isinstance(a, Lam): return _eq(a.body, b.body)
    if isinstance(a, Pi): return _eq(a.domain, b.domain) and _eq(a.codomain, b.codomain)
    if isinstance(a, Sigma): return _eq(a.fst_type, b.fst_type) and _eq(a.snd_type, b.snd_type)
    if isinstance(a, Pair): return _eq(a.fst, b.fst) and _eq(a.snd, b.snd)
    if isinstance(a, Fst): return _eq(a.pair, b.pair)
    if isinstance(a, Snd): return _eq(a.pair, b.pair)
    if isinstance(a, Sum): return _eq(a.left, b.left) and _eq(a.right, b.right)
    if isinstance(a, Inl): return _eq(a.term, b.term)
    if isinstance(a, Inr): return _eq(a.term, b.term)
    if isinstance(a, Case): return _eq(a.scrutinee, b.scrutinee) and _eq(a.left_branch, b.left_branch) and _eq(a.right_branch, b.right_branch)
    if isinstance(a, Id): return _eq(a.ty, b.ty) and _eq(a.lhs, b.lhs) and _eq(a.rhs, b.rhs)
    if isinstance(a, Atom): return a.name == b.name
    if isinstance(a, Universe): return a.sort == b.sort
    if isinstance(a, (Refl, Nat, Zero, Bottom)): return True
    if isinstance(a, Suc): return _eq(a.n, b.n)
    return False


# ── Type Checking ────────────────────────────────────────────────────

def infer(ctx: Context, term: Term) -> Term:
    if isinstance(term, Var):
        ty = context_lookup(ctx, term.index)
        if ty is None:
            raise TypeCheckError(f"Variable v{term.index} out of bounds (context has {len(ctx)} entries)")
        return ty

    elif isinstance(term, App):
        func_ty = normalize(infer(ctx, term.func))
        if not isinstance(func_ty, Pi):
            raise TypeCheckError(f"Application: function has type {func_ty}, expected Π-type")
        check(ctx, term.arg, func_ty.domain)
        return normalize(subst_top(func_ty.codomain, term.arg))

    elif isinstance(term, Fst):
        pair_ty = normalize(infer(ctx, term.pair))
        if not isinstance(pair_ty, Sigma):
            raise TypeCheckError(f"Fst: expected Σ-type, got {pair_ty}")
        return pair_ty.fst_type

    elif isinstance(term, Snd):
        pair_ty = normalize(infer(ctx, term.pair))
        if not isinstance(pair_ty, Sigma):
            raise TypeCheckError(f"Snd: expected Σ-type, got {pair_ty}")
        return normalize(subst_top(pair_ty.snd_type, Fst(term.pair)))

    elif isinstance(term, Case):
        scrut_ty = normalize(infer(ctx, term.scrutinee))
        if not isinstance(scrut_ty, Sum):
            raise TypeCheckError(f"Case: expected Sum type, got {scrut_ty}")
        left_ty = normalize(infer(ctx, term.left_branch))
        if not isinstance(left_ty, Pi):
            raise TypeCheckError(f"Case: left branch has type {left_ty}, expected Π")
        if not terms_equal(left_ty.domain, scrut_ty.left):
            raise TypeCheckError(f"Case: left domain {left_ty.domain} ≠ {scrut_ty.left}")
        right_ty = normalize(infer(ctx, term.right_branch))
        if not isinstance(right_ty, Pi):
            raise TypeCheckError(f"Case: right branch has type {right_ty}, expected Π")
        if not terms_equal(right_ty.domain, scrut_ty.right):
            raise TypeCheckError(f"Case: right domain {right_ty.domain} ≠ {scrut_ty.right}")
        if not terms_equal(left_ty.codomain, right_ty.codomain):
            raise TypeCheckError(f"Case: branch codomains differ")
        return left_ty.codomain

    elif isinstance(term, Nat):
        return Universe(Set(0))

    elif isinstance(term, Zero):
        return Nat()

    elif isinstance(term, Suc):
        check(ctx, term.n, Nat())
        return Nat()

    elif isinstance(term, Universe):
        if isinstance(term.sort, Set):
            return Universe(Set(term.sort.level + 1))
        return Universe(Set(0))

    elif isinstance(term, Atom):
        return Universe(Set(0))

    elif isinstance(term, Bottom):
        return Universe(Set(0))

    elif isinstance(term, Absurd):
        # absurd : ⊥ → A  (can produce any type, but we need check mode for target type)
        check(ctx, term.proof, Bottom())
        raise TypeCheckError("Absurd: cannot infer result type — use check mode")

    elif isinstance(term, Pi):
        return Universe(Set(0))

    elif isinstance(term, Sigma):
        return Universe(Set(0))

    elif isinstance(term, Sum):
        return Universe(Set(0))

    elif isinstance(term, Id):
        return Universe(Set(0))

    elif isinstance(term, (Lam, Pair, Inl, Inr, Refl)):
        raise TypeCheckError(f"Cannot infer type of {type(term).__name__} — use check mode")

    raise TypeCheckError(f"Cannot infer type of: {term}")


def check(ctx: Context, term: Term, expected: Term) -> bool:
    expected = normalize(expected)

    if isinstance(term, Lam):
        if not isinstance(expected, Pi):
            raise TypeCheckError(f"Lambda checked against non-Π type {expected}")
        new_ctx = context_extend(ctx, expected.domain)
        check(new_ctx, term.body, expected.codomain)
        return True

    elif isinstance(term, Pair):
        if not isinstance(expected, Sigma):
            raise TypeCheckError(f"Pair checked against non-Σ type {expected}")
        check(ctx, term.fst, expected.fst_type)
        snd_expected = normalize(subst_top(expected.snd_type, term.fst))
        check(ctx, term.snd, snd_expected)
        return True

    elif isinstance(term, Absurd):
        # Ex falso: from ⊥ derive anything
        check(ctx, term.proof, Bottom())
        return True  # any expected type — ⊥ proves everything

    elif isinstance(term, Inl):
        if not isinstance(expected, Sum):
            raise TypeCheckError(f"Inl checked against non-Sum type {expected}")
        check(ctx, term.term, expected.left)
        return True

    elif isinstance(term, Inr):
        if not isinstance(expected, Sum):
            raise TypeCheckError(f"Inr checked against non-Sum type {expected}")
        check(ctx, term.term, expected.right)
        return True

    elif isinstance(term, Refl):
        if not isinstance(expected, Id):
            raise TypeCheckError(f"Refl checked against non-Id type {expected}")
        if not terms_equal(expected.lhs, expected.rhs):
            raise TypeCheckError(f"Refl: {expected.lhs} ≠ {expected.rhs}")
        return True

    else:
        inferred = normalize(infer(ctx, term))
        if not terms_equal(inferred, expected):
            raise TypeCheckError(f"Type mismatch: inferred {inferred}, expected {expected}")
        return True


def gate(ctx: Context, term: Term, goal: Term) -> bool:
    try:
        return check(ctx, term, goal)
    except TypeCheckError:
        return False


# ── Proof optimality ─────────────────────────────────────────────────

def term_size(term: Term) -> int:
    """Count the number of constructors in a term."""
    if isinstance(term, Var):
        return 1
    elif isinstance(term, App):
        return 1 + term_size(term.func) + term_size(term.arg)
    elif isinstance(term, Lam):
        return 1 + term_size(term.body)
    elif isinstance(term, Pair):
        return 1 + term_size(term.fst) + term_size(term.snd)
    elif isinstance(term, (Fst, Snd)):
        return 1 + term_size(term.pair)
    elif isinstance(term, (Inl, Inr)):
        return 1 + term_size(term.term)
    elif isinstance(term, Case):
        return 1 + term_size(term.scrutinee) + term_size(term.left_branch) + term_size(term.right_branch)
    elif isinstance(term, Absurd):
        return 1 + term_size(term.proof)
    elif isinstance(term, Suc):
        return 1 + term_size(term.n)
    elif isinstance(term, (Refl, Zero, Nat, Bottom, Atom, Universe)):
        return 1
    return 1


def proof_optimality(term: Term) -> float:
    """Compute optimality ratio: normalized_size / raw_size.

    1.0 = proof is already optimal (normalization doesn't shrink it).
    <1.0 = proof has redundant steps.

    Reward signal: how efficiently did the model find the proof?
    """
    raw = term_size(term)
    normalized = normalize(term)
    norm_size = term_size(normalized)
    if raw == 0:
        return 1.0
    return min(norm_size / raw, 1.0)  # cap at 1.0 (normalization can't grow)


# ── Structured feedback ──────────────────────────────────────────────

@dataclass
class GateFeedback:
    valid: bool
    error: str = ""
    expected_type: object = None
    inferred_type: object = None
    position: str = ""
    hint: str = ""


def gate_with_feedback(ctx: Context, term: Term, goal: Term) -> GateFeedback:
    try:
        check(ctx, term, goal)
        return GateFeedback(valid=True)
    except TypeCheckError as e:
        msg = str(e)
        fb = GateFeedback(valid=False, error=msg, expected_type=goal)
        if "Type mismatch" in msg:
            fb.hint = f"Need type {goal}, got wrong type"
            try:
                fb.inferred_type = infer(ctx, term)
            except TypeCheckError:
                pass
        elif "out of bounds" in msg:
            fb.hint = f"Variable index too large. Context has {len(ctx)} entries"
        elif "expected Π" in msg or "non-Π" in msg:
            fb.hint = "Need Π-type (function type)"
        elif "non-Σ" in msg:
            fb.hint = "Need Σ-type (pair type)"
        elif "non-Sum" in msg:
            fb.hint = "Need Sum type (disjunction)"
        if ctx:
            available = [f"v{i}:{ctx[i]}" for i in range(min(len(ctx), 10))]
            fb.position = f"Available: {', '.join(available)}"
        return fb
