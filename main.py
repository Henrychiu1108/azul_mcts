from azul_game import GameState
from mcts import MCTS, Node
# from alpha_beta import get_best_move_alpha_beta  # 暫停使用 Alpha-Beta
import random
import sys

# 移除原本的整體狀態 dump，改為情境化輸出

def mcts(game):
    root = Node(game.copy())
    m = MCTS(root)
    m.search(iterations=100)
    return m.get_best_move()

# 暫停使用 Alpha-Beta 與最後回合判定
# def alpha_beta(game_state, depth=10):
#     best_move = get_best_move_alpha_beta(game_state, depth)
#     return best_move
#
# def is_last_round(game) -> bool:
#     return any(
#         sum(player.board.occupancy[row]) == 4
#         for player in game.players
#         for row in range(5)
#     )

def main():
    original_stdout = sys.stdout
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        game = GameState()
        game.refill_factories()
        round_number = 1
        turn_in_round = 0
        print(f"=== Round {round_number} Start ===")

        while True:
            legal_moves = game.get_legal_moves()
            if not legal_moves:  # 回合結束
                print(f"--- Round {round_number} End ---")
                # 顯示分數
                for idx, p in enumerate(game.players):
                    print(f"Player {idx} Score: {p.score}")
                game.end_round()
                if game.is_terminal():
                    break
                round_number += 1
                turn_in_round = 0
                print(f"\n=== Round {round_number} Start ===")
                continue

            turn_in_round += 1
            print(f"Turn {turn_in_round} - Player {game.current_player}")
            best_move = mcts(game)
            game.make_move(best_move)
            game.switch_player()

        # 遊戲結束最終分數
        print("\n=== Game End ===")
        for idx, p in enumerate(game.players):
            print(f"Player {idx} Final Score: {p.score}")
        winner = game.get_winner()
        print(f"Winner: Player {winner}" if winner is not None else "Winner: Tie")
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
