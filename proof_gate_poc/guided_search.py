"""
Type-guided proof search — constrained decoding.

Instead of generating the whole proof and checking at the end,
the type checker guides EACH step:

1. Model outputs a constructor (APP, LAM, VAR_i, ...)
2. Type checker says what types are needed next (subgoals)
3. Invalid tokens are masked — model can only choose valid options
4. Repeat until proof is complete or no valid options remain

This is a tactic-style prover where the transformer is the tactic engine
and the type checker maintains the proof state.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .ir import (
    Type, Atom, Arrow, Pi, Sigma, Sum,
    Term, Var, App, Lam, Pair, Fst, Snd, Inl, Inr, Case,
    Context, context_lookup, context_extend,
)
from .type_checker import infer, check, TypeCheckError


@dataclass
class ProofGoal:
    """A subgoal in the proof state."""
    ctx: Context    # current context (premises)
    goal: Type      # type to inhabit


@dataclass
class SearchState:
    """State of the guided proof search."""
    goals: List[ProofGoal]      # stack of subgoals to solve
    partial_term: object        # term being built (with holes)
    complete: bool = False
    term: Optional[Term] = None # the complete term (when done)


def get_valid_actions(ctx: Context, goal: Type, lemma_bank=None) -> List[dict]:
    """Given a context and goal type, return all valid first-step actions.

    Includes lemmas from the lemma bank as available actions.

    Each action is a dict with:
      - 'action': action type
      - 'subgoals': list of new ProofGoal
      - 'description': human-readable
      - 'builds' or 'builds_partial': how to construct the term
    """
    from .type_checker import terms_equal

    actions = []

    # VAR_i: if any variable in context has the goal type, we're done
    for i, ty in enumerate(ctx):
        if ty == goal:
            actions.append({
                'action': 'var',
                'index': i,
                'subgoals': [],
                'description': f'v{i} : {ty} (direct)',
                'builds': Var(i),
            })

    # APP: if any variable has type X → goal, we can apply it
    # subgoal: prove X
    for i, ty in enumerate(ctx):
        if isinstance(ty, Pi) and ty.codomain == goal:
            actions.append({
                'action': 'app',
                'func_index': i,
                'subgoals': [ProofGoal(ctx=ctx, goal=ty.domain)],
                'description': f'apply v{i} : {ty}, need to prove {ty.domain}',
                'builds_partial': ('app_var', i),
            })

    # APP on deeper arrows: if variable has type A → B → goal, etc.
    for i, ty in enumerate(ctx):
        if isinstance(ty, Pi) and isinstance(ty.codomain, Pi) and ty.codomain.codomain == goal:
            actions.append({
                'action': 'app2',
                'func_index': i,
                'subgoals': [
                    ProofGoal(ctx=ctx, goal=ty.domain),
                    ProofGoal(ctx=ctx, goal=ty.codomain.domain),
                ],
                'description': f'apply v{i} : {ty} twice, need {ty.domain} and {ty.codomain.domain}',
                'builds_partial': ('app2_var', i),
            })

    # LAM: if goal is Arrow(A, B), introduce A and prove B
    if isinstance(goal, Pi):
        new_ctx = context_extend(ctx, goal.domain)
        actions.append({
            'action': 'lam',
            'subgoals': [ProofGoal(ctx=new_ctx, goal=goal.codomain)],
            'description': f'assume {goal.domain}, prove {goal.codomain}',
            'builds_partial': 'lam',
        })

    # PAIR: if goal is Prod(A, B), prove both A and B
    if isinstance(goal, Sigma):
        actions.append({
            'action': 'pair',
            'subgoals': [
                ProofGoal(ctx=ctx, goal=goal.fst_type),
                ProofGoal(ctx=ctx, goal=goal.snd_type),
            ],
            'description': f'prove both {goal.fst_type} and {goal.snd_type}',
            'builds_partial': 'pair',
        })

    # FST/SND: if any variable has product type containing goal
    for i, ty in enumerate(ctx):
        if isinstance(ty, Sigma):
            if ty.fst_type == goal:
                actions.append({
                    'action': 'fst',
                    'index': i,
                    'subgoals': [],
                    'description': f'fst of v{i} : {ty}',
                    'builds': Fst(Var(i)),
                })
            if ty.snd_type == goal:
                actions.append({
                    'action': 'snd',
                    'index': i,
                    'subgoals': [],
                    'description': f'snd of v{i} : {ty}',
                    'builds': Snd(Var(i)),
                })

    # APP + FST/SND: extract arrow from product then apply
    for i, ty in enumerate(ctx):
        if isinstance(ty, Sigma):
            if isinstance(ty.fst_type, Pi) and ty.fst_type.codomain == goal:
                actions.append({
                    'action': 'app_fst',
                    'index': i,
                    'subgoals': [ProofGoal(ctx=ctx, goal=ty.fst_type.domain)],
                    'description': f'apply fst(v{i}) : {ty.fst_type}, need {ty.fst_type.domain}',
                    'builds_partial': ('app_fst', i),
                })
            if isinstance(ty.snd_type, Pi) and ty.snd_type.codomain == goal:
                actions.append({
                    'action': 'app_snd',
                    'index': i,
                    'subgoals': [ProofGoal(ctx=ctx, goal=ty.snd_type.domain)],
                    'description': f'apply snd(v{i}) : {ty.snd_type}, need {ty.snd_type.domain}',
                    'builds_partial': ('app_snd', i),
                })

    # INL: if goal is Sum(A, B), prove A
    if isinstance(goal, Sum):
        actions.append({
            'action': 'inl',
            'subgoals': [ProofGoal(ctx=ctx, goal=goal.left)],
            'description': f'left injection, prove {goal.left}',
            'builds_partial': 'inl',
        })
        actions.append({
            'action': 'inr',
            'subgoals': [ProofGoal(ctx=ctx, goal=goal.right)],
            'description': f'right injection, prove {goal.right}',
            'builds_partial': 'inr',
        })

    # CASE: if any variable has sum type, do case analysis
    for i, ty in enumerate(ctx):
        if isinstance(ty, Sum):
            actions.append({
                'action': 'case',
                'index': i,
                'subgoals': [
                    ProofGoal(ctx=ctx, goal=Arrow(ty.left, goal)),
                    ProofGoal(ctx=ctx, goal=Arrow(ty.right, goal)),
                ],
                'description': f'case on v{i} : {ty}, prove {ty.left}→{goal} and {ty.right}→{goal}',
                'builds_partial': ('case', i),
            })

    # ── LEMMA ACTIONS: use proven theorems as one-step solutions ────
    if lemma_bank is not None:
        for lem_idx in range(lemma_bank.size()):
            entry = lemma_bank.get_by_index(lem_idx)
            if entry is None:
                continue
            lem_premises, lem_goal, lem_term = entry

            # Direct match: lemma proves exactly our goal with no premises
            if not lem_premises and terms_equal(lem_goal, goal):
                actions.append({
                    'action': 'lemma',
                    'lemma_index': lem_idx,
                    'subgoals': [],
                    'description': f'lem{lem_idx}: ⊢ {lem_goal} (proven)',
                    'builds': lem_term,
                })

            # Lemma with premises: if all premises are available in context
            elif lem_premises and terms_equal(lem_goal, goal):
                # Check if we can satisfy all lemma premises from context
                premise_subgoals = []
                for lp in lem_premises:
                    # lp is stored as string after persistence — try matching
                    lp_str = str(lp)
                    found = False
                    for ci, ct in enumerate(ctx):
                        if str(ct) == lp_str:
                            found = True
                            break
                    if not found:
                        premise_subgoals.append(
                            ProofGoal(ctx=ctx, goal=lp if isinstance(lp, Term) else goal)
                        )

                if not premise_subgoals:
                    # All premises satisfied — lemma applies directly
                    actions.append({
                        'action': 'lemma',
                        'lemma_index': lem_idx,
                        'subgoals': [],
                        'description': f'lem{lem_idx}: {lem_premises} ⊢ {lem_goal} (all premises in ctx)',
                        'builds': lem_term,
                    })

            # Lemma as function: lemma proves X → goal, we need to prove X
            if isinstance(lem_goal, str):
                # After persistence, lem_goal is string — skip functional matching
                continue
            if hasattr(lem_goal, 'codomain') and not lem_premises:
                # Lemma proves A → B. If B = goal, subgoal is A
                if isinstance(lem_goal, Pi) and hasattr(lem_goal, 'codomain') and terms_equal(lem_goal.codomain, goal):
                    actions.append({
                        'action': 'lemma_app',
                        'lemma_index': lem_idx,
                        'subgoals': [ProofGoal(ctx=ctx, goal=lem_goal.domain)],
                        'description': f'apply lem{lem_idx}: {lem_goal}, need {lem_goal.domain}',
                        'builds_partial': ('lemma_app', lem_idx),
                        '_lem_term': lem_term,
                    })

    return actions


def build_term(action: dict, subterms: List[Term]) -> Term:
    """Build a term from an action and its solved subgoals."""
    if 'builds' in action:
        return action['builds']

    partial = action.get('builds_partial')

    if partial == 'lam':
        return Lam(subterms[0])
    elif partial == 'pair':
        return Pair(subterms[0], subterms[1])
    elif partial == 'inl':
        return Inl(subterms[0])
    elif partial == 'inr':
        return Inr(subterms[0])
    elif isinstance(partial, tuple):
        kind = partial[0]
        i = partial[1]
        if kind == 'app_var':
            return App(Var(i), subterms[0])
        elif kind == 'app2_var':
            return App(App(Var(i), subterms[0]), subterms[1])
        elif kind == 'app_fst':
            return App(Fst(Var(i)), subterms[0])
        elif kind == 'app_snd':
            return App(Snd(Var(i)), subterms[0])
        elif kind == 'case':
            return Case(Var(i), subterms[0], subterms[1])
        elif kind == 'lemma_app':
            # Apply lemma term to the solved subgoal
            lem_idx = partial[1]
            # We need the lemma term — get from action
            return App(action.get('_lem_term', Var(0)), subterms[0])

    raise ValueError(f"Cannot build term for action: {action}")
