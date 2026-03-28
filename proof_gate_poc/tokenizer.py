"""
Unified Tokenizer — bidirectional map between token IDs and typed lambda calculus IR.

Covers STLC (propositional) + dependent types + identity + naturals + universes.
Token = Term constructor. The vocabulary IS the term language.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from .ir import (
    Term, Var, App, Lam,
    Pi, Sigma, Pair, Fst, Snd,
    Sum, Inl, Inr, Case,
    Id, Refl, J,
    Nat, Zero, Suc, NatElim,
    Bottom, Absurd,
    Universe, Set, Prop, Atom,
)

# ── Vocabulary ───────────────────────────────────────────────────────
# MINIMAL BASIS. Everything else is derived as lemmas.
#
# Primitives:    VAR, APP, LAM, PI, SET — the core of typed λ-calculus
# Lemma ref:     LEMMA — pointer into lemma bank (proven theorems)
# Structure:     PAD, BOS, EOS, SEP, GOAL
# Atoms:         A-Z — named base types for convenience
#
# Everything else (Σ, ∨, Id, ℕ, Pair, Fst, Case, Refl, ...) can be
# defined through Π + Set and stored as lemmas. The model derives them.

MAX_VARS = 32
MAX_ATOMS = 26
MAX_LEVELS = 4
MAX_LEMMAS = 1024   # lemma bank slots (pointer network extends this)

# Special
PAD = 0
BOS = 1
EOS = 2
SEP = 3
GOAL = 4

# ── PRIMITIVES (5 constructors — the irreducible basis) ─────────────
C_APP = 5           # application
C_LAM = 6           # lambda abstraction
T_PI = 7            # Π-type (dependent function — subsumes →)
T_SET_BASE = 8      # Set₀=8, Set₁=9, Set₂=10, Set₃=11
T_ATOM_BASE = 12    # A=12, B=13, ..., Z=37

# ── DERIVED (built from primitives, but kept as tokens for efficiency) ──
# These CAN be encoded via church/scott encoding through Π+Set+λ,
# but having them as tokens makes proof search tractable.
# Model learns to use them; they're equivalent to proven lemmas.
T_SIGMA = 38        # Σ-type (derived: Σ(A,B) = Π(C:Set). (Π(A). Π(B). C) → C)
T_SUM = 39          # coproduct (derived via church encoding)
T_ID = 40           # identity type
T_NAT = 41          # natural numbers
C_PAIR = 42
C_FST = 43
C_SND = 44
C_INL = 45
C_INR = 46
C_CASE = 47
C_REFL = 48
C_J = 49
C_ZERO = 50
C_SUC = 51
C_NATELIM = 52
T_BOTTOM = 53       # ⊥
C_ABSURD = 54       # ex falso

# ── LEMMA REFERENCE — pointer into lemma bank ───────────────────────
C_LEMMA_BASE = 55   # LEMMA₀=53, ..., LEMMA₁₀₂₃=1076

# ── VARIABLES (pointer network, but also fixed tokens as fallback) ──
C_VAR_BASE = C_LEMMA_BASE + MAX_LEMMAS  # 1077..1108

VOCAB_SIZE = C_VAR_BASE + MAX_VARS  # 1109

# Backward compat
T_ARROW = T_PI
T_PROD = T_SIGMA
T_PROP_BASE = T_SET_BASE  # simplification: Prop = Set for now

# Token classification
PRIMITIVE_TOKENS = {C_APP, C_LAM, T_PI, T_BOTTOM} | set(range(T_SET_BASE, T_SET_BASE + MAX_LEVELS))
DERIVED_TOKENS = {T_SIGMA, T_SUM, T_ID, T_NAT, C_PAIR, C_FST, C_SND,
                  C_INL, C_INR, C_CASE, C_REFL, C_J, C_ZERO, C_SUC, C_NATELIM, C_ABSURD}
LEMMA_TOKENS = set(range(C_LEMMA_BASE, C_LEMMA_BASE + MAX_LEMMAS))
VAR_TOKENS = set(range(C_VAR_BASE, C_VAR_BASE + MAX_VARS))
TERM_TOKENS = PRIMITIVE_TOKENS | DERIVED_TOKENS | LEMMA_TOKENS | VAR_TOKENS

# Human-readable names
TOKEN_NAMES = {
    PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", SEP: "<sep>", GOAL: "<goal>",
    C_APP: "app", C_LAM: "λ", T_PI: "Π",
    T_SIGMA: "Σ", T_SUM: "∨", T_ID: "Id", T_NAT: "ℕ",
    C_PAIR: "pair", C_FST: "π₁", C_SND: "π₂",
    C_INL: "inl", C_INR: "inr", C_CASE: "case",
    C_REFL: "refl", C_J: "J", C_ZERO: "zero", C_SUC: "suc", C_NATELIM: "natrec",
    T_BOTTOM: "⊥", C_ABSURD: "absurd",
}
for i in range(MAX_ATOMS):
    TOKEN_NAMES[T_ATOM_BASE + i] = chr(ord('A') + i)
for i in range(MAX_VARS):
    TOKEN_NAMES[C_VAR_BASE + i] = f"v{i}"
for i in range(MAX_LEVELS):
    TOKEN_NAMES[T_SET_BASE + i] = f"Set{i}" if i > 0 else "Set"
for i in range(MAX_LEMMAS):
    TOKEN_NAMES[C_LEMMA_BASE + i] = f"lem{i}"


def token_name(tok: int) -> str:
    return TOKEN_NAMES.get(tok, f"?{tok}")


def tokens_to_str(tokens: List[int]) -> str:
    return " ".join(token_name(t) for t in tokens)


# ── Serialization: Term → Tokens ─────────────────────────────────────

def type_to_tokens(ty: Term) -> List[int]:
    """Serialize a type to tokens (prefix notation). Alias for term_to_tokens."""
    return term_to_tokens(ty)


def term_to_tokens(term: Term) -> List[int]:
    """Serialize any term/type to tokens in prefix notation."""
    if isinstance(term, Var):
        return [C_VAR_BASE + min(term.index, MAX_VARS - 1)]
    elif isinstance(term, App):
        return [C_APP] + term_to_tokens(term.func) + term_to_tokens(term.arg)
    elif isinstance(term, Lam):
        return [C_LAM] + term_to_tokens(term.body)
    elif isinstance(term, Pi):
        return [T_PI] + term_to_tokens(term.domain) + term_to_tokens(term.codomain)
    elif isinstance(term, Sigma):
        return [T_SIGMA] + term_to_tokens(term.fst_type) + term_to_tokens(term.snd_type)
    elif isinstance(term, Pair):
        return [C_PAIR] + term_to_tokens(term.fst) + term_to_tokens(term.snd)
    elif isinstance(term, Fst):
        return [C_FST] + term_to_tokens(term.pair)
    elif isinstance(term, Snd):
        return [C_SND] + term_to_tokens(term.pair)
    elif isinstance(term, Sum):
        return [T_SUM] + term_to_tokens(term.left) + term_to_tokens(term.right)
    elif isinstance(term, Inl):
        return [C_INL] + term_to_tokens(term.term)
    elif isinstance(term, Inr):
        return [C_INR] + term_to_tokens(term.term)
    elif isinstance(term, Case):
        return [C_CASE] + term_to_tokens(term.scrutinee) + term_to_tokens(term.left_branch) + term_to_tokens(term.right_branch)
    elif isinstance(term, Id):
        return [T_ID] + term_to_tokens(term.ty) + term_to_tokens(term.lhs) + term_to_tokens(term.rhs)
    elif isinstance(term, Refl):
        return [C_REFL]
    elif isinstance(term, J):
        return [C_J] + term_to_tokens(term.ty) + term_to_tokens(term.x) + term_to_tokens(term.motive) + term_to_tokens(term.base) + term_to_tokens(term.y) + term_to_tokens(term.proof)
    elif isinstance(term, Nat):
        return [T_NAT]
    elif isinstance(term, Zero):
        return [C_ZERO]
    elif isinstance(term, Suc):
        return [C_SUC] + term_to_tokens(term.n)
    elif isinstance(term, NatElim):
        return [C_NATELIM] + term_to_tokens(term.motive) + term_to_tokens(term.base) + term_to_tokens(term.step) + term_to_tokens(term.n)
    elif isinstance(term, Bottom):
        return [T_BOTTOM]
    elif isinstance(term, Absurd):
        return [C_ABSURD] + term_to_tokens(term.proof)
    elif isinstance(term, Universe):
        if isinstance(term.sort, Set):
            return [T_SET_BASE + min(term.sort.level, MAX_LEVELS - 1)]
        elif isinstance(term.sort, Prop):
            return [T_PROP_BASE + min(term.sort.level, MAX_LEVELS - 1)]
    elif isinstance(term, Atom):
        idx = ord(term.name) - ord('A')
        if 0 <= idx < MAX_ATOMS:
            return [T_ATOM_BASE + idx]
        return [T_ATOM_BASE]  # fallback
    raise ValueError(f"Cannot tokenize: {term}")


# ── Deserialization: Tokens → Term ───────────────────────────────────

def tokens_to_term(tokens: List[int], pos: int = 0) -> Tuple[Optional[Term], int]:
    """Deserialize tokens to a term."""
    if pos >= len(tokens):
        return None, pos
    tok = tokens[pos]
    if tok == EOS:
        return None, pos

    # Variables
    if C_VAR_BASE <= tok < C_VAR_BASE + MAX_VARS:
        return Var(tok - C_VAR_BASE), pos + 1

    # Term constructors
    elif tok == C_APP:
        f, pos = tokens_to_term(tokens, pos + 1)
        if f is None: return None, pos
        a, pos = tokens_to_term(tokens, pos)
        if a is None: return None, pos
        return App(f, a), pos
    elif tok == C_LAM:
        b, pos = tokens_to_term(tokens, pos + 1)
        if b is None: return None, pos
        return Lam(b), pos
    elif tok == C_PAIR:
        a, pos = tokens_to_term(tokens, pos + 1)
        if a is None: return None, pos
        b, pos = tokens_to_term(tokens, pos)
        if b is None: return None, pos
        return Pair(a, b), pos
    elif tok == C_FST:
        p, pos = tokens_to_term(tokens, pos + 1)
        if p is None: return None, pos
        return Fst(p), pos
    elif tok == C_SND:
        p, pos = tokens_to_term(tokens, pos + 1)
        if p is None: return None, pos
        return Snd(p), pos
    elif tok == C_INL:
        t, pos = tokens_to_term(tokens, pos + 1)
        if t is None: return None, pos
        return Inl(t), pos
    elif tok == C_INR:
        t, pos = tokens_to_term(tokens, pos + 1)
        if t is None: return None, pos
        return Inr(t), pos
    elif tok == C_CASE:
        s, pos = tokens_to_term(tokens, pos + 1)
        if s is None: return None, pos
        l, pos = tokens_to_term(tokens, pos)
        if l is None: return None, pos
        r, pos = tokens_to_term(tokens, pos)
        if r is None: return None, pos
        return Case(s, l, r), pos
    elif tok == C_REFL:
        return Refl(), pos + 1
    elif tok == C_ZERO:
        return Zero(), pos + 1
    elif tok == C_SUC:
        n, pos = tokens_to_term(tokens, pos + 1)
        if n is None: return None, pos
        return Suc(n), pos
    elif tok == C_J:
        ty, pos = tokens_to_term(tokens, pos + 1)
        if ty is None: return None, pos
        x, pos = tokens_to_term(tokens, pos)
        if x is None: return None, pos
        m, pos = tokens_to_term(tokens, pos)
        if m is None: return None, pos
        b, pos = tokens_to_term(tokens, pos)
        if b is None: return None, pos
        y, pos = tokens_to_term(tokens, pos)
        if y is None: return None, pos
        p, pos = tokens_to_term(tokens, pos)
        if p is None: return None, pos
        return J(ty, x, m, b, y, p), pos
    elif tok == T_BOTTOM:
        return Bottom(), pos + 1
    elif tok == C_ABSURD:
        p, pos = tokens_to_term(tokens, pos + 1)
        if p is None: return None, pos
        return Absurd(p), pos
    elif tok == C_NATELIM:
        m, pos = tokens_to_term(tokens, pos + 1)
        if m is None: return None, pos
        b, pos = tokens_to_term(tokens, pos)
        if b is None: return None, pos
        s, pos = tokens_to_term(tokens, pos)
        if s is None: return None, pos
        n, pos = tokens_to_term(tokens, pos)
        if n is None: return None, pos
        return NatElim(m, b, s, n), pos

    # Type constructors
    elif tok == T_PI:
        d, pos = tokens_to_term(tokens, pos + 1)
        if d is None: return None, pos
        c, pos = tokens_to_term(tokens, pos)
        if c is None: return None, pos
        return Pi(d, c), pos
    elif tok == T_SIGMA:
        a, pos = tokens_to_term(tokens, pos + 1)
        if a is None: return None, pos
        b, pos = tokens_to_term(tokens, pos)
        if b is None: return None, pos
        return Sigma(a, b), pos
    elif tok == T_SUM:
        a, pos = tokens_to_term(tokens, pos + 1)
        if a is None: return None, pos
        b, pos = tokens_to_term(tokens, pos)
        if b is None: return None, pos
        return Sum(a, b), pos
    elif tok == T_ID:
        ty, pos = tokens_to_term(tokens, pos + 1)
        if ty is None: return None, pos
        l, pos = tokens_to_term(tokens, pos)
        if l is None: return None, pos
        r, pos = tokens_to_term(tokens, pos)
        if r is None: return None, pos
        return Id(ty, l, r), pos
    elif tok == T_NAT:
        return Nat(), pos + 1
    elif T_SET_BASE <= tok < T_SET_BASE + MAX_LEVELS:
        return Universe(Set(tok - T_SET_BASE)), pos + 1
    elif T_PROP_BASE <= tok < T_PROP_BASE + MAX_LEVELS:
        return Universe(Prop(tok - T_PROP_BASE)), pos + 1
    elif T_ATOM_BASE <= tok < T_ATOM_BASE + MAX_ATOMS:
        return Atom(chr(ord('A') + (tok - T_ATOM_BASE))), pos + 1

    return None, pos


# ── Problem encoding ─────────────────────────────────────────────────

def encode_problem(premises: List[Term], goal: Term) -> List[int]:
    """Encode premises + goal as input tokens."""
    tokens = [BOS]
    for i, p in enumerate(premises):
        if i > 0:
            tokens.append(SEP)
        tokens.extend(term_to_tokens(p))
    tokens.append(GOAL)
    tokens.extend(term_to_tokens(goal))
    tokens.append(EOS)
    return tokens


def decode_problem(tokens: List[int]) -> Tuple[List[Term], Term]:
    """Decode input tokens back to premises + goal."""
    pos = 1 if tokens[0] == BOS else 0
    premises = []
    while pos < len(tokens) and tokens[pos] != GOAL:
        if tokens[pos] == SEP:
            pos += 1
            continue
        ty, pos = tokens_to_term(tokens, pos)
        if ty is not None:
            premises.append(ty)
    pos += 1  # skip GOAL
    goal, pos = tokens_to_term(tokens, pos)
    return premises, goal
