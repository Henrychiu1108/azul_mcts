import math
import random
from azul_game import GameState

class Node:
    def __init__(self, state: GameState, parent=None, move=None, root_player=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = move
        # 未嘗試之合法手列表（增量擴展）
        self.untried_moves = state.get_legal_moves()
        # 回合結束（來源全取完，尚未 end_round）的評分快取
        self.round_end_reward = None
        # 根節點記錄 root_player；子節點繼承
        if root_player is not None:
            self.root_player = root_player
        else:
            self.root_player = parent.root_player if parent else state.current_player

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.4):
        choices_weights = [
            (child.value / child.visits if child.visits > 0 else 0) + c * math.sqrt(2 * math.log(max(1, self.visits)) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        if not self.untried_moves:
            return None
        move = self.untried_moves.pop()
        new_state = self.state.copy()  # 需要：建立子節點的專屬狀態
        new_state.make_move(move)
        legal_after = new_state.get_legal_moves()  # 只呼叫一次
        if legal_after:
            new_state.switch_player()
        child_node = Node(new_state, self, move)
        if not legal_after:  # 回合末預先計算 reward（純計算，不呼叫 end_round、不再 copy）
            child_node.round_end_reward = MCTS._round_end_reward_for_state(new_state, child_node.root_player)
        self.children.append(child_node)
        return child_node

class MCTS:
    def __init__(self, root: Node):
        self.root = root
        self.root_player = root.state.current_player

    # 純計算：給定『來源已耗盡、尚未真正 end_round』的狀態，評估結束後雙方暫時分數差
    @staticmethod
    def _round_end_reward_for_state(state: GameState, root_player: int) -> int:
        provisional = []
        for pid, player in enumerate(state.players):
            placements = state._compute_full_line_placements(player)
            gain = 0
            for _, _, g, _ in placements:
                gain += g
            penalty = state._floor_penalty_value(player.floor)
            score_after = player.score + gain + penalty
            if score_after < 0:
                score_after = 0
            provisional.append(score_after)
        return provisional[root_player] - provisional[1 - root_player]

    def search(self, iterations=50000):
        for _ in range(iterations):
            node = self.select(self.root)
            expanded = node.expand()
            if expanded and expanded.round_end_reward is not None:
                # 直接回傳快取 reward，跳過 simulate
                self.backpropagate(expanded, expanded.round_end_reward)
                continue
            node_to_sim = expanded if expanded else node
            reward = self.simulate(node_to_sim)
            self.backpropagate(node_to_sim, reward)

    def select(self, node: Node):
        while node.is_fully_expanded():
            if not node.state.get_legal_moves():
                break
            node = node.best_child()
        return node

    def simulate(self, node: Node):
        # 仍需 copy：模擬過程會改動狀態，但不能影響樹節點
        state = node.state.copy()

        def heuristic_score(move):
            dest_type = move[3]
            if dest_type == 'floor':
                return 0.1
            if dest_type == 'pattern_line':
                dest_index = move[4]
                color = move[2]
                player = state.players[state.current_player]
                column = None
                for col in range(5):
                    if player.board.pattern[dest_index, col] == color:
                        column = col
                        break
                if column is not None:
                    wall = player.board.occupancy
                    if wall[dest_index][column] == 0:
                        temp_wall = wall.copy()
                        temp_wall[dest_index][column] = 1
                        if all(temp_wall[dest_index]):
                            return 20
                        if all(temp_wall[i][column] for i in range(5)):
                            return 20
                        for row in range(5):
                            if sum(wall[row]) == 4 and wall[row][column] == 0:
                                return 15
                return 1 + (4 - dest_index)
            return 1

        while True:
            moves = state.get_legal_moves()
            if not moves:
                break
            scores = [heuristic_score(m) for m in moves]
            move = random.choices(moves, weights=scores, k=1)[0]
            state.make_move(move)
            if state.get_legal_moves():
                state.switch_player()

        # 不再呼叫 end_round() 與額外 copy；直接純計算回合結束 reward
        reward = self._round_end_reward_for_state(state, self.root_player)
        return reward

    def backpropagate(self, node: Node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_best_move(self):
        if not self.root.children:
            return None
        best_child = self.root.best_child(c=0)
        return best_child.move
