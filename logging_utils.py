import csv, os

MOVES_CSV_PATH = 'moves_log.csv'

_def_header_written = False

# 新增：重置 CSV（每局覆寫）

def reset_move_csv():
    global _def_header_written
    if os.path.exists(MOVES_CSV_PATH):
        try:
            os.remove(MOVES_CSV_PATH)
        except OSError:
            pass
    _def_header_written = False

# --- Tile / board rendering helpers ---

def _color_letter(tile_or_color):
    from azul_game import Color, FirstPlayerMarker  # lazy import to avoid circular at module load
    if tile_or_color is None:
        return '.'
    if isinstance(tile_or_color, FirstPlayerMarker):
        return '★'
    color = getattr(tile_or_color, 'color', tile_or_color)
    mapping = {
        Color.BLUE: 'U',
        Color.BLACK: 'K',
        Color.RED: 'R',
        Color.GREEN: 'G',
        Color.YELLOW: 'Y'
    }
    return mapping.get(color, '?')


def _print_player_board(player, prefix='    '):
    # Only wall (no pattern lines) for compactness
    rows = []
    for r in range(5):
        row = []
        for c in range(5):
            if player.board.occupancy[r, c] == 1:
                row.append(_color_letter(player.board.pattern[r, c]))
            else:
                row.append('.')
        rows.append(prefix + ' '.join(row))
    return '\n'.join(rows)

# --- CSV header handling ---

def _ensure_csv_header(heuristic_names):
    global _def_header_written
    if _def_header_written and os.path.exists(MOVES_CSV_PATH):
        return
    need_header = not os.path.exists(MOVES_CSV_PATH)
    if need_header:
        with open(MOVES_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            base_cols = [
                'round', 'turn', 'player',
                'source_type', 'source_index', 'color', 'dest_type', 'dest_row',
                'total', 'visits', 'value', 'avg'
            ]
            writer.writerow(base_cols + heuristic_names)
    _def_header_written = True

# --- Public logging functions for game_output.txt (no per-move spam) ---

def log_round_start(round_number, game, out_stream=None):
    print(f"=== Round {round_number} Start State ===", file=out_stream or None)
    for pid, player in enumerate(game.players):
        print(f"Player {pid} Score: {player.score}", file=out_stream or None)
        print(_print_player_board(player), file=out_stream or None)
        print('-', file=out_stream or None)


def log_round_end(round_number, summaries, game, out_stream=None):
    print(f"=== Round {round_number} End State ===", file=out_stream or None)
    # summaries: list of (gain, penalty, pre_score, post_score)
    for pid, (gain, penalty, pre_score, post_score) in enumerate(summaries):
        net = gain + penalty
        print(f"Player {pid}: +{gain} {penalty} (net {net}) -> {pre_score} -> {post_score}", file=out_stream or None)
        print(_print_player_board(game.players[pid]), file=out_stream or None)
        print('-', file=out_stream or None)


def log_final_board(game, out_stream=None):
    print("=== Final Board State ===", file=out_stream or None)
    for pid, player in enumerate(game.players):
        print(f"Player {pid} Board:", file=out_stream or None)
        print(_print_player_board(player), file=out_stream or None)
        print(f"Player {pid} Final Score: {player.score}", file=out_stream or None)
        print('-', file=out_stream or None)

# --- CSV move logging (still used; per-move textual logging removed from stdout) ---

def write_move_csv(round_number, turn_index, player, move_info):
    mv = move_info['move']
    source_type, source_index, color_obj, dest_type, dest_row = mv
    color_str = getattr(color_obj, 'value', str(color_obj))
    dest_row_val = dest_row if dest_type == 'pattern_line' else ''
    heur_dict = {name: val for name, val in move_info['details']}
    heuristic_names = [name for name, _ in move_info['details']]
    _ensure_csv_header(heuristic_names)
    row = [
        round_number, turn_index, player,
        source_type, source_index if source_index is not None else '', color_str,
        dest_type, dest_row_val,
        f"{move_info['total']:.4f}", move_info['visits'], f"{move_info['value']:.4f}", f"{move_info['avg']:.4f}"
    ]
    for hn in heuristic_names:
        row.append(f"{heur_dict.get(hn,0.0):.4f}")
    with open(MOVES_CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(row)

# Optional legacy function (kept if you want to quickly re-enable per-move text logging)

def log_move_text(round_number, turn_index, current_player, move_info, out_stream=None):
    mv = move_info['move']
    total = move_info['total']
    details = ', '.join(f"{name}={val:.2f}" for name, val in move_info['details'])
    print(f"[R{round_number} T{turn_index}] P{current_player} {mv} total={total:.3f} | {details}", file=out_stream or None)
