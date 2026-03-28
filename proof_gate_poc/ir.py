"""
Unified Typed Lambda Calculus IR.

One IR for everything: propositional logic, dependent types, HoTT.
STLC is a subset — Arrow is Pi without dependency, Prod is Sigma without dependency.

Under Curry-Howard:
  Type  = Proposition
  Term  = Proof
  Γ ⊢ t : A  =  "t is a valid proof of A from assumptions Γ"

Mirrors Agda's Internal syntax (Agda.Syntax.Internal):
  Var, App, Lam, Pi, Sigma, Pair, Fst, Snd, Sum, Inl, Inr, Case,
  Id, Refl, J, Nat, Zero, Suc, NatElim, Universe
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Tuple, Optional


# ── Core: Types and Terms are the same thing ─────────────────────────
# In dependent types there is no separation. A type IS a term of type Set.

@dataclass(frozen=True)
class Var:
    """Variable (de Bruijn index)."""
    index: int
    def __repr__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class App:
    """Application (modus ponens)."""
    func: Term
    arg: Term
    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


@dataclass(frozen=True)
class Lam:
    """Lambda abstraction (→ / Π introduction)."""
    body: Term
    def __repr__(self) -> str:
        return f"(λ.{self.body})"


# ── Function types ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Pi:
    """Π-type: dependent function. Π(x:A).B(x).
    When B doesn't reference x, this is just A → B (Arrow)."""
    domain: Term
    codomain: Term  # may reference Var(0) for dependency
    def __repr__(self) -> str:
        return f"(Π {self.domain} . {self.codomain})"


def Arrow(domain: Term, codomain: Term) -> Pi:
    """Non-dependent function type. A → B = Π(_:A).B"""
    return Pi(domain, codomain)


# ── Pair types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sigma:
    """Σ-type: dependent pair. Σ(x:A).B(x).
    When B doesn't reference x, this is just A ∧ B (Prod)."""
    fst_type: Term
    snd_type: Term  # may reference Var(0) for dependency
    def __repr__(self) -> str:
        return f"(Σ {self.fst_type} . {self.snd_type})"


def Prod(left: Term, right: Term) -> Sigma:
    """Non-dependent pair type. A ∧ B = Σ(_:A).B"""
    return Sigma(left, right)


@dataclass(frozen=True)
class Pair:
    """Pair constructor (Σ introduction / ∧ introduction)."""
    fst: Term
    snd: Term
    def __repr__(self) -> str:
        return f"⟨{self.fst}, {self.snd}⟩"


@dataclass(frozen=True)
class Fst:
    """First projection (Σ elimination / ∧ elimination left)."""
    pair: Term
    def __repr__(self) -> str:
        return f"π₁({self.pair})"


@dataclass(frozen=True)
class Snd:
    """Second projection (Σ elimination / ∧ elimination right)."""
    pair: Term
    def __repr__(self) -> str:
        return f"π₂({self.pair})"


# ── Sum types (disjunction) ──────────────────────────────────────────

@dataclass(frozen=True)
class Sum:
    """Sum type / coproduct. A ∨ B."""
    left: Term
    right: Term
    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Inl:
    """Left injection (∨ introduction left)."""
    term: Term
    def __repr__(self) -> str:
        return f"inl({self.term})"


@dataclass(frozen=True)
class Inr:
    """Right injection (∨ introduction right)."""
    term: Term
    def __repr__(self) -> str:
        return f"inr({self.term})"


@dataclass(frozen=True)
class Case:
    """Case analysis (∨ elimination)."""
    scrutinee: Term
    left_branch: Term
    right_branch: Term
    def __repr__(self) -> str:
        return f"case({self.scrutinee}, {self.left_branch}, {self.right_branch})"


# ── Identity / Path types ────────────────────────────────────────────

@dataclass(frozen=True)
class Id:
    """Identity type. Id A a b = "a equals b in type A"."""
    ty: Term
    lhs: Term
    rhs: Term
    def __repr__(self) -> str:
        return f"(Id {self.ty} {self.lhs} {self.rhs})"


@dataclass(frozen=True)
class Refl:
    """Reflexivity: proof that a = a."""
    def __repr__(self) -> str:
        return "refl"


@dataclass(frozen=True)
class J:
    """J eliminator for identity types."""
    ty: Term
    x: Term
    motive: Term
    base: Term
    y: Term
    proof: Term
    def __repr__(self) -> str:
        return f"(J {self.ty} {self.x} {self.motive} {self.base} {self.y} {self.proof})"


# ── Natural numbers ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Nat:
    """Natural number type."""
    def __repr__(self) -> str:
        return "ℕ"


@dataclass(frozen=True)
class Zero:
    """zero : ℕ"""
    def __repr__(self) -> str:
        return "zero"


@dataclass(frozen=True)
class Suc:
    """suc : ℕ → ℕ"""
    n: Term
    def __repr__(self) -> str:
        return f"(suc {self.n})"


@dataclass(frozen=True)
class NatElim:
    """Dependent eliminator for ℕ."""
    motive: Term
    base: Term
    step: Term
    n: Term
    def __repr__(self) -> str:
        return f"(natrec {self.motive} {self.base} {self.step} {self.n})"


# ── Universes ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Set:
    """Universe: Set ℓ"""
    level: int = 0
    def __repr__(self) -> str:
        return f"Set{self.level}" if self.level > 0 else "Set"


@dataclass(frozen=True)
class Prop:
    """Universe of mere propositions."""
    level: int = 0
    def __repr__(self) -> str:
        return f"Prop{self.level}" if self.level > 0 else "Prop"


@dataclass(frozen=True)
class Universe:
    """Sort as a term."""
    sort: Union[Set, Prop]
    def __repr__(self) -> str:
        return repr(self.sort)


# ── Bottom (empty type / absurdity) ──────────────────────────────────

@dataclass(frozen=True)
class Bottom:
    """⊥ — the empty type. No constructors. No inhabitants.
    ¬A = A → ⊥ (negation is function to bottom)."""
    def __repr__(self) -> str:
        return "⊥"


@dataclass(frozen=True)
class Absurd:
    """Ex falso quodlibet: from ⊥ prove anything.
    absurd : ⊥ → A"""
    proof: Term  # proof of ⊥
    def __repr__(self) -> str:
        return f"absurd({self.proof})"


def Neg(ty: Term) -> Pi:
    """Negation: ¬A = A → ⊥"""
    return Pi(ty, Bottom())


# ── Atom (convenience alias for named base types in simple problems) ──

@dataclass(frozen=True)
class Atom:
    """Named atomic type. A, B, C, ... — sugar for propositional variables."""
    name: str
    def __repr__(self) -> str:
        return self.name


# ── Term union ───────────────────────────────────────────────────────

Term = Union[
    Var, App, Lam,                          # core λ-calculus
    Pi, Sigma, Pair, Fst, Snd,              # dependent function + pair
    Sum, Inl, Inr, Case,                    # coproduct
    Id, Refl, J,                            # identity / path
    Nat, Zero, Suc, NatElim,               # natural numbers
    Bottom, Absurd,                         # absurdity / negation
    Universe, Atom,                         # universes + atoms
]

# For backward compatibility
Type = Term


# ── Context ──────────────────────────────────────────────────────────

Context = Tuple[Term, ...]


def context_lookup(ctx: Context, index: int) -> Optional[Term]:
    if 0 <= index < len(ctx):
        return ctx[index]
    return None


def context_extend(ctx: Context, ty: Term) -> Context:
    return (ty,) + ctx
