from azul_game import GameState
from mcts import MCTS, Node
import sys
from logging_utils import (
    write_move_csv,
    reset_move_csv,
    log_turn_start,
    log_pre_scoring_state,
    log_post_round_scores,
    log_final_scores,
)

def mcts(game):
    root = Node(game.copy())
    m = MCTS(root)
    m.search(iterations=300)
    breakdowns = m.explain_moves(root.state)  # 已含排名與指標
    best = None
    best_key = -1e9
    for info in breakdowns:
        if info['visits'] > 0:
            key = info['avg']
        else:
            key = info['total'] * 0.001
        if key > best_key:
            best_key = key
            best = info
    return best

def main():
    original_stdout = sys.stdout
    reset_move_csv()
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        game = GameState()
        game.refill_factories()
        round_number = 1
        turn_index = 1

        while True:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                # 回合計分前狀態（清線前）
                log_pre_scoring_state(round_number, game)
                pre_infos = []  # (gain, penalty, pre_score)
                for pid, player in enumerate(game.players):
                    placements = game._compute_full_line_placements(player)
                    gain = sum(g for _, _, g, _ in placements)
                    penalty = game._floor_penalty_value(player.floor)
                    pre_infos.append((gain, penalty, player.score))
                game.end_round()
                summaries = []
                for (gain, penalty, pre_score), player in zip(pre_infos, game.players):
                    summaries.append((gain, penalty, pre_score, player.score))
                log_post_round_scores(round_number, summaries)
                if game.is_terminal():
                    break
                round_number += 1
                turn_index = 1
                # 下一回合第一手前輸出工廠狀態
                log_turn_start(round_number, turn_index, game)
                continue
            # 有合法手：先輸出該手開始的工廠狀態
            log_turn_start(round_number, turn_index, game)
            move_info = mcts(game)
            write_move_csv(round_number, turn_index, game.current_player, move_info)
            best_move = move_info['move']
            game.make_move(best_move)
            game.switch_player()
            turn_index += 1

        winner = game.get_winner()
        log_final_scores(game)
        print(f"Winner: Player {winner}" if winner is not None else "Winner: Tie")
    sys.stdout = original_stdout
    print("Output written to game_output.txt and moves_log.csv")

if __name__ == "__main__":
    main()
