from azul_game import GameState, Color, FirstPlayerMarker
from mcts import MCTS, Node
import sys

# 只保留必要：回合結束時輸出玩家牆、pattern lines、floor、分數

def _color_letter(tile_or_color):
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

# 新增：結算前輸出 pattern lines 與 floor

def print_round_pre_settlement(game: GameState, round_number: int):
    print(f"=== Round {round_number} Pre-Settlement ===")
    for pid, player in enumerate(game.players):
        print(f"Player {pid} Pre-Score: {player.score}")
        print(f"Player {pid} Pattern Lines:")
        for line_idx in range(5):
            line = player.pattern_lines.lines[line_idx]
            cap = line_idx + 1
            filled = ''.join(_color_letter(t) for t in line)
            padded = filled + '.' * (cap - len(line))
            print(f"    L{line_idx} [{padded}]")
        floor_line = ' '.join(_color_letter(t) for t in player.floor.tiles)
        print(f"Player {pid} Floor: {floor_line}")
        print('-')

# 修改：結算後僅輸出 Board（牆）與新分數

def print_round_post_settlement(game: GameState, round_number: int):
    print(f"=== Round {round_number} Result ===")
    for pid, player in enumerate(game.players):
        print(f"Player {pid} Board:")
        for r in range(5):
            row = []
            for c in range(5):
                if player.board.occupancy[r, c] == 1:
                    row.append(_color_letter(player.board.pattern[r, c]))
                else:
                    row.append('.')
            print('    ' + ' '.join(row))
        print(f"Player {pid} Score: {player.score}")
        print('-')

def mcts(game):
    root = Node(game.copy())
    m = MCTS(root)
    m.search(iterations=100)
    return m.get_best_move()

def main():
    original_stdout = sys.stdout
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        game = GameState()
        game.refill_factories()
        round_number = 1

        while True:
            legal_moves = game.get_legal_moves()
            if not legal_moves:  # 回合結束：先印 pattern lines / floor，再結算，再印牆
                print_round_pre_settlement(game, round_number)
                game.end_round()
                print_round_post_settlement(game, round_number)
                if game.is_terminal():
                    break
                round_number += 1
                continue
            best_move = mcts(game)
            game.make_move(best_move)
            game.switch_player()

        print("=== Game End ===")
        for idx, p in enumerate(game.players):
            print(f"Player {idx} Final Score: {p.score}")
        winner = game.get_winner()
        print(f"Winner: Player {winner}" if winner is not None else "Winner: Tie")
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
