from azul_game import GameState, Color, FirstPlayerMarker
from mcts import MCTS, Node
import sys

# 簡化：只需結算回合時的得失分與最終棋盤

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

# 回合摘要：每位玩家本回合放置所得 (gain) 與地板懲罰 (penalty)，以及淨變化

def log_round_summary(round_number, player_summaries):
    print(f"=== Round {round_number} Summary ===")
    for pid, data in enumerate(player_summaries):
        gain, penalty, pre_score, post_score = data
        net = gain + penalty
        print(f"Player {pid}: +{gain} {penalty} (net {net}) -> {pre_score} -> {post_score}")
    print('-')

# 最終棋盤與分數

def log_final_board(game: GameState):
    print("=== Final Board State ===")
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
        print(f"Player {pid} Final Score: {player.score}")
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
            if not legal_moves:
                # 回合結束前預先計算每位玩家的放置 gain 與地板 penalty
                pre_infos = []  # (gain, penalty, pre_score)
                for pid, player in enumerate(game.players):
                    placements = game._compute_full_line_placements(player)
                    gain = sum(g for _, _, g, _ in placements)
                    penalty = game._floor_penalty_value(player.floor)
                    pre_infos.append((gain, penalty, player.score))
                game.end_round()
                # 取結束後分數
                summaries = []
                for (gain, penalty, pre_score), player in zip(pre_infos, game.players):
                    summaries.append((gain, penalty, pre_score, player.score))
                log_round_summary(round_number, summaries)
                if game.is_terminal():
                    break
                round_number += 1
                continue
            best_move = mcts(game)
            game.make_move(best_move)
            game.switch_player()

        # 遊戲結束計入終局加分並輸出最終棋盤與分數
        winner = game.get_winner()  # 內部會加終局 bonus
        log_final_board(game)
        print(f"Winner: Player {winner}" if winner is not None else "Winner: Tie")
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
