"""
Hole-based proof state — data structures for iterative proof construction.

The model fills holes one at a time. Type checker maintains the proof state.
"""
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass, field

from .ir import Type, Term, Context
from .guided_search import get_valid_actions, build_term


@dataclass
class Hole:
    """A hole in the proof — a subgoal to fill."""
    ctx: Context
    goal: Type
    id: int


@dataclass
class ProofState:
    """Current state of iterative proof construction."""
    holes: List[Hole]
    tree: dict = field(default_factory=dict)
    next_hole_id: int = 1
    root_hole_id: int = 0


def init_proof_state(ctx: Context, goal: Type) -> ProofState:
    return ProofState(holes=[Hole(ctx=ctx, goal=goal, id=0)])


def fill_hole(state: ProofState, hole_idx: int, action: dict) -> ProofState:
    """Fill a hole with an action, creating new holes for subgoals."""
    hole = state.holes[hole_idx]
    child_ids = []
    new_holes = []
    for sg in action['subgoals']:
        h = Hole(ctx=sg.ctx, goal=sg.goal, id=state.next_hole_id)
        child_ids.append(state.next_hole_id)
        new_holes.append(h)
        state.next_hole_id += 1

    state.tree[hole.id] = (action, child_ids)
    remaining = [h for i, h in enumerate(state.holes) if i != hole_idx]
    remaining.extend(new_holes)
    state.holes = remaining
    return state


def reconstruct_term(state: ProofState, hole_id: int = 0) -> Optional[Term]:
    """Reconstruct the complete term from the proof tree."""
    if hole_id not in state.tree:
        return None

    action, child_ids = state.tree[hole_id]

    if 'builds' in action:
        return action['builds']

    subterms = []
    for cid in child_ids:
        t = reconstruct_term(state, cid)
        if t is None:
            return None
        subterms.append(t)

    return build_term(action, subterms)
