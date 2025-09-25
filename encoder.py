"""Raw state encoding (v1.2c-B) for Azul MCTS -> NN integration.

Factory encoding scheme C (compressed counts):
- Each factory encoded as 5 color counts normalized by 4 (capacity) => 5 * 5 = 25 dims total.
- Removes per-slot order (not strategically relevant). Retains exact per-color multiplicity.

Pattern line scheme B (compressed per line):
- Each of 5 pattern lines encoded as: 5-color one-hot (line color OR all zeros if empty) + fill_ratio (len/capacity).
- Per player: 5 * 6 = 30 dims (vs previous 90). Both players: 60.

Dimensions breakdown (per perspective):
  Wall occupancy (both players):                       50
  Pattern lines (both players, compressed):            60
  Floors (both players) -> counts /7:                  2
  Factories (5 factories * 5 color counts /4):         25
  Center (color counts /20 + marker):                  6
  Bag remaining color distribution:                    5
  Discard color distribution:                          5
  Scores (/100):                                       2
  Current player flag:                                 1
  First player marker holder one-hot:                  2
Total = 50+60+2+25+6+5+5+2+1+2 = 158

encode_state(state, perspective) -> float32[158]
Perspective only affects ordering of "player 0/1" in paired features.
"""
from __future__ import annotations
from typing import List
import numpy as np
from azul_game import GameState, Color, bag_color_counts, discard_color_counts

COLORS: List[Color] = list(Color)
NUM_COLORS = len(COLORS)

ENCODING_SIZE = 158

# Helper constants
FACTORY_COUNT = 5
FACTORY_CAPACITY = 4
FLOOR_SLOTS = 7
PATTERN_LINE_SLOTS = [1,2,3,4,5]  # capacities
TOTAL_PATTERN_SLOTS = sum(PATTERN_LINE_SLOTS)  # 15 (kept for reference)

# One-hot helpers

def _one_hot(index: int, size: int) -> List[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]

# Encoding functions for components

def _encode_wall(player) -> List[float]:
    return player.board.occupancy.flatten().astype(float).tolist()

def _encode_pattern_lines(player) -> List[float]:
    """Compressed per-line representation (Scheme B):
    For each of the 5 lines:
      - 5 color one-hot (all zeros if empty)
      - 1 fill_ratio = len(line)/capacity (0..1)
    Total 6 * 5 = 30 per player."""
    out: List[float] = []
    for line_idx, cap in enumerate(PATTERN_LINE_SLOTS):
        line = player.pattern_lines.lines[line_idx]
        if line:  # has tiles
            color_idx = COLORS.index(line[0].color)
            color_vec = _one_hot(color_idx, NUM_COLORS)
            fill_ratio = len(line) / cap
        else:
            color_vec = [0.0] * NUM_COLORS
            fill_ratio = 0.0
        out.extend(color_vec)
        out.append(fill_ratio)
    return out

def _encode_factories(state: GameState) -> List[float]:
    """Compressed per-factory color counts (normalized by 4). 5 factories * 5 colors = 25 dims.
    Empty factory => all zeros for that 5-color block."""
    out: List[float] = []
    for f_idx in range(FACTORY_COUNT):
        tiles = state.factories[f_idx].tiles
        if tiles:
            counts = {c:0 for c in COLORS}
            for t in tiles:
                counts[t.color] += 1
            for c in COLORS:
                out.append(counts[c] / 4.0)
        else:
            out.extend([0.0]*NUM_COLORS)
    return out

def _encode_center(state: GameState) -> List[float]:
    counts = {c:0 for c in COLORS}
    marker_flag = 1.0 if state.center.has_first_player_marker else 0.0
    for t in state.center.tiles:
        counts[t.color] += 1
    # Normalize by 20 (initial total per color in bag) for stability
    out = [counts[c] / 20.0 for c in COLORS]
    out.append(marker_flag)
    return out

def _encode_bag(state: GameState) -> List[float]:
    counts = bag_color_counts(state)
    total = sum(counts.values()) or 1
    return [counts[c] / total for c in COLORS]

def _encode_discard(state: GameState) -> List[float]:
    counts = discard_color_counts(state)
    total = sum(counts.values()) or 1
    return [counts[c] / total for c in COLORS]

def encode_state(state: GameState, perspective: int) -> np.ndarray:
    assert perspective in (0,1)
    me = state.players[perspective]
    opp = state.players[1 - perspective]

    feats: List[float] = []

    # 1. Walls
    feats.extend(_encode_wall(me))
    feats.extend(_encode_wall(opp))

    # 2. Pattern lines (compressed)
    feats.extend(_encode_pattern_lines(me))
    feats.extend(_encode_pattern_lines(opp))

    # 3. Floors (counts only)
    floor_count_me = sum(1 for t in me.floor.tiles if t is not None)
    floor_count_opp = sum(1 for t in opp.floor.tiles if t is not None)
    feats.append(floor_count_me / FLOOR_SLOTS)
    feats.append(floor_count_opp / FLOOR_SLOTS)

    # 4. Factories (compressed)
    feats.extend(_encode_factories(state))

    # 5. Center (color counts /20 + marker)
    feats.extend(_encode_center(state))

    # 6. Bag + Discard
    feats.extend(_encode_bag(state))
    feats.extend(_encode_discard(state))

    # 7. Scores
    feats.append(me.score / 100.0)
    feats.append(opp.score / 100.0)

    # 8. Current player flag
    feats.append(1.0 if state.current_player == perspective else 0.0)

    # 9. First player marker holder one-hot
    holder = state.first_player_marker_holder
    feats.append(1.0 if holder == perspective else 0.0)
    feats.append(1.0 if holder == (1 - perspective) else 0.0)

    arr = np.asarray(feats, dtype=np.float32)
    assert arr.shape[0] == ENCODING_SIZE, f"Unexpected length {arr.shape[0]} != {ENCODING_SIZE}"
    return arr

__all__ = ["encode_state", "ENCODING_SIZE"]
