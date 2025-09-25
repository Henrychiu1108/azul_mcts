"""Action space definition for Azul (fixed 180 actions).

Encoding schema:
  sources (6): factory0..factory4, center(5)
  colors  (5): order = COLORS list from Color enum
  destinations (6): pattern_line0..pattern_line4, floor(5)
Action id formula:
  id = ((source_idx * NUM_COLORS) + color_idx) * NUM_DESTINATIONS + dest_idx
  where:
    source_idx in [0..5] (0-4 factories, 5 center)
    color_idx  in [0..4]
    dest_idx   in [0..5] (0-4 pattern lines, 5 floor)

Move tuple format (consistent with GameState.get_legal_moves):
  (source_type, source_index, color, destination_type, line_index)
  source_type: 'factory' | 'center' ; source_index: int or None
  destination_type: 'pattern_line' | 'floor' ; line_index: int or None
"""
from typing import Optional, Tuple
import numpy as np
from azul_game import Color, GameState

COLORS: list[Color] = list(Color)
COLOR_TO_INDEX = {c: i for i, c in enumerate(COLORS)}
NUM_FACTORIES = 5
NUM_SOURCES = NUM_FACTORIES + 1  # factories + center
NUM_COLORS = len(COLORS)         # 5
NUM_PATTERN_LINES = 5
NUM_DESTINATIONS = NUM_PATTERN_LINES + 1  # + floor
ACTION_SPACE_SIZE = NUM_SOURCES * NUM_COLORS * NUM_DESTINATIONS  # 6*5*6 = 180

# ---------------------------------------------------------------------------
# Encoding / decoding (no defensive try/except for speed; assume validated inputs)
# ---------------------------------------------------------------------------

def encode_action(source_type: str, source_index: Optional[int], color: Color,
                  destination_type: str, line_index: Optional[int]) -> int:
    if source_type == 'factory':
        source_idx = source_index  # assume 0..4
    else:  # center
        source_idx = NUM_FACTORIES
    color_idx = COLOR_TO_INDEX[color]
    if destination_type == 'pattern_line':
        dest_idx = line_index  # assume 0..4
    else:  # floor
        dest_idx = NUM_PATTERN_LINES
    return ((source_idx * NUM_COLORS) + color_idx) * NUM_DESTINATIONS + dest_idx

def decode_action(action_id: int) -> Tuple[str, Optional[int], Color, str, Optional[int]]:
    # Keep range check (cheap) to avoid silent corruption
    if not (0 <= action_id < ACTION_SPACE_SIZE):
        raise ValueError(f"action_id out of range: {action_id}")
    src_color_block, dest_idx = divmod(action_id, NUM_DESTINATIONS)
    source_idx, color_idx = divmod(src_color_block, NUM_COLORS)
    source_type = 'center' if source_idx == NUM_FACTORIES else 'factory'
    source_index = None if source_type == 'center' else source_idx
    color = COLORS[color_idx]
    if dest_idx == NUM_PATTERN_LINES:
        return (source_type, source_index, color, 'floor', None)
    return (source_type, source_index, color, 'pattern_line', dest_idx)

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def move_to_action_id(move: Tuple[str, int, Color, str, int]) -> int:
    source_type, source_index, color, destination_type, line_index = move
    return encode_action(source_type, source_index, color, destination_type, line_index)

def moves_to_action_ids(moves: list[Tuple[str, int, Color, str, int]]) -> list[int]:
    return [move_to_action_id(m) for m in moves]

# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def legal_action_mask(state: GameState, perspective: int) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    if perspective != state.current_player:
        return mask
    for mv in state.get_legal_moves():
        aid = move_to_action_id(mv)
        mask[aid] = 1.0
    return mask

__all__ = [
    'COLORS', 'COLOR_TO_INDEX', 'NUM_FACTORIES', 'NUM_SOURCES', 'NUM_COLORS', 'NUM_PATTERN_LINES',
    'NUM_DESTINATIONS', 'ACTION_SPACE_SIZE', 'encode_action', 'decode_action',
    'move_to_action_id', 'moves_to_action_ids', 'legal_action_mask'
]
