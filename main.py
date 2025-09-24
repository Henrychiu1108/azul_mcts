from azul_game import GameState
from mcts import MCTS, Node
import sys
from logging_utils import write_move_csv, log_round_start, log_round_end, log_final_board, reset_move_csv

def mcts(game):
    root = Node(game.copy())
    m = MCTS(root)
    m.search(iterations=300)
    # 對根節點狀態做解釋
    breakdowns = m.explain_moves(root.state)
    # 選擇最高平均 value 的子（若無 visits 以 heuristic total 排）
    # 與 get_best_move 不同：這裡直接計算，並回傳附帶說明
    best = None
    best_key = -1e9
    for info in breakdowns:
        if info['visits'] > 0:
            key = info['avg']
        else:
            key = info['total'] * 0.001  # 尚未模擬的略降權
        if key > best_key:
            best_key = key
            best = info
    return best

def main():
    original_stdout = sys.stdout
    reset_move_csv()
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        print('Move CSV: moves_log.csv')
        game = GameState()
        game.refill_factories()
        round_number = 1
        turn_index = 1
        log_round_start(round_number, game)

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
                log_round_end(round_number, summaries, game)
                if game.is_terminal():
                    break
                round_number += 1
                turn_index = 1
                log_round_start(round_number, game)
                continue
            move_info = mcts(game)
            write_move_csv(round_number, turn_index, game.current_player, move_info)
            best_move = move_info['move']
            game.make_move(best_move)
            game.switch_player()
            turn_index += 1

        # 遊戲結束計入終局加分並輸出最終棋盤與分數
        winner = game.get_winner()  # 內部會加終局 bonus
        log_final_board(game)
        print(f"Winner: Player {winner}" if winner is not None else "Winner: Tie")
    sys.stdout = original_stdout
    print("Output written to game_output.txt and moves_log.csv")

if __name__ == "__main__":
    main()
