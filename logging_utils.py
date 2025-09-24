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
    from azul_game import Color, FirstPlayerMarker  # lazy import
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


def _print_wall_only(player, prefix='    '):
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


def _print_pattern_lines(player, prefix='    '):
    # Each pattern line: show filled tiles then remaining capacity as '.'
    lines_out = []
    for r, line in enumerate(player.pattern_lines.lines):
        cap = r + 1
        letters = ''.join(_color_letter(t) for t in line)
        pad = '.' * (cap - len(line))
        lines_out.append(f"{prefix}L{r}:{letters}{pad}")
    return '\n'.join(lines_out)


def _print_floor(player, prefix='    '):
    tiles = ''.join(_color_letter(t) for t in player.floor.tiles if t is not None)
    empties = 7 - sum(1 for t in player.floor.tiles if t is not None)
    return f"{prefix}F:{tiles}{'.'*empties}"

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
            extra_metric_cols = [
                'heuristic_rank', 'q_rank', 'rank_diff', 'spearman', 'top_match', 'num_moves'
            ]
            writer.writerow(base_cols + heuristic_names + extra_metric_cols)
    _def_header_written = True

# --- Public logging functions required ---

def log_turn_start(round_number, turn_index, game, out_stream=None):
    # Factories + center only
    print(f"[Round {round_number} Turn {turn_index}] Factories", file=out_stream or None)
    # Factories
    for i, fac in enumerate(game.factories):
        letters = ''.join(_color_letter(t) for t in fac.tiles)
        print(f"  F{i}: {letters}", file=out_stream or None)
    # Center
    center_letters = ''.join(_color_letter(t) for t in game.center.tiles)
    print(f"  C : {center_letters}", file=out_stream or None)
    print('-', file=out_stream or None)


def log_pre_scoring_state(round_number, game, out_stream=None):
    print(f"=== Round {round_number} Pre-Scoring State ===", file=out_stream or None)
    for pid, player in enumerate(game.players):
        print(f"Player {pid} Score(before): {player.score}", file=out_stream or None)
        print(_print_wall_only(player), file=out_stream or None)
        print(_print_pattern_lines(player), file=out_stream or None)
        print(_print_floor(player), file=out_stream or None)
        print('-', file=out_stream or None)


def log_post_round_scores(round_number, summaries, out_stream=None):
    # summaries: (gain, penalty, pre_score, post_score)
    print(f"=== Round {round_number} Scores ===", file=out_stream or None)
    for pid, (gain, penalty, pre_score, post_score) in enumerate(summaries):
        net = gain + penalty
        print(f"Player {pid}: +{gain} {penalty} (net {net}) -> {pre_score} -> {post_score}", file=out_stream or None)
    print('-', file=out_stream or None)


def log_final_scores(game, out_stream=None):
    print("=== Final Scores ===", file=out_stream or None)
    for pid, p in enumerate(game.players):
        print(f"Player {pid}: {p.score}", file=out_stream or None)
    print('-', file=out_stream or None)

# --- CSV move logging ---

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
    # 附加排名與對齊指標（若缺則空白）
    h_rank = move_info.get('heuristic_rank', '')
    q_rank = move_info.get('q_rank', '')
    rank_diff = ''
    if isinstance(h_rank, int) and isinstance(q_rank, int):
        rank_diff = q_rank - h_rank
    spearman = move_info.get('spearman', '')
    top_match = move_info.get('top_match', '')
    num_moves = move_info.get('num_moves', '')
    row.extend([h_rank, q_rank, rank_diff, spearman if spearman != None else '', top_match, num_moves])
    with open(MOVES_CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(row)
