import math
import random
from azul_game import GameState  # 移除未使用 FirstPlayerMarker
from heuristics import HeuristicMixin

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

class MCTS(HeuristicMixin):
    # 僅保留與 MCTS 搜尋策略相關的參數
    ROLLOUT_TEMPERATURE = 1.2
    ROLLOUT_UNIFORM_MIX = 0.15

    def __init__(self, root: Node, heuristics=None, use_heuristics: bool = True):
        self.root = root
        self.root_player = root.state.current_player
        self.use_heuristics = use_heuristics  # 新增：可暫時停用 heuristic
        # 預設使用 mixin 中的 heuristics 方法
        self.heuristics = heuristics or [
            self.h_floor_action,
            self.h_first_player_marker,
            self.h_line_progress,
            self.h_ev_row,
            self.h_ev_col,
            self.h_adjacency
        ]
        self.weights = [1.0] * len(self.heuristics)

    # 仍需：供 heuristics 呼叫的上下文抽取
    def _extract_move_context(self, state: GameState, move):
        dest_type = move[3]
        if dest_type != 'pattern_line':
            return (False, None, None, None, None, None)
        dest_row = move[4]
        color = move[2]
        player = state.players[state.current_player]
        column = player.board.row_color_to_col[dest_row][color]
        assert player.board.occupancy[dest_row, column] == 0, "Heuristic context: position already occupied"
        return (True, player, dest_row, color, column, player.board.occupancy)

    # Heuristic 拆解 / 說明
    def heuristic_breakdown(self, state: GameState, move):
        details = []
        raw_vals = []
        total_weighted = 0.0
        for i, h in enumerate(self.heuristics):
            try:
                val = h(state, move)
            except Exception:
                val = 0.0
            details.append((h.__name__, val))
            raw_vals.append(val)
            w = self.weights[i]
            total_weighted += w * val
        return total_weighted, details, raw_vals

    def explain_moves(self, state: GameState):
        moves = state.get_legal_moves()
        out = []
        child_map = {}
        if self.root:
            for c in self.root.children:
                child_map[c.move] = c
        for mv in moves:
            total, details, raw = self.heuristic_breakdown(state, mv)
            node = child_map.get(mv)
            visits = node.visits if node else 0
            value = node.value if node else 0.0
            avg = (value / visits) if visits > 0 else 0.0
            out.append({
                'move': mv,
                'total': total,
                'details': details,
                'raw': raw,
                'visits': visits,
                'value': value,
                'avg': avg
            })
        if not out:
            return out
        sorted_h = sorted(out, key=lambda x: x['total'], reverse=True)
        sorted_q = sorted(out, key=lambda x: (x['avg'] if x['visits']>0 else -1e9), reverse=True)
        h_rank_map = {id(m): i for i, m in enumerate(sorted_h)}
        q_rank_map = {id(m): i for i, m in enumerate(sorted_q)}
        visited = [m for m in out if m['visits']>0]
        spearman = ''
        if len(visited) >= 2:
            d2 = 0
            n = len(visited)
            for m in visited:
                d = h_rank_map[id(m)] - q_rank_map[id(m)]
                d2 += d*d
            spearman_val = 1 - 6*d2/(n*(n*n-1)) if n>1 else 0.0
            spearman = f"{spearman_val:.4f}"
        top_match = ''
        if sorted_h and sorted_q:
            top_match = int(sorted_h[0]['move'] == sorted_q[0]['move'])
        num_moves = len(out)
        for m in out:
            m['heuristic_rank'] = h_rank_map[id(m)]
            m['q_rank'] = q_rank_map[id(m)]
            m['spearman'] = spearman
            m['top_match'] = top_match
            m['num_moves'] = num_moves
        return out

    # 聚合 heuristic 分數
    def _aggregate_heuristic(self, state: GameState, move):
        total = 0.0
        for w, h in zip(self.weights, self.heuristics):
            try:
                v = h(state, move)
            except Exception:
                v = 0.0
            total += w * v
        return total

    # softmax 權重計算
    def _softmax_weights(self, scores, temperature):
        if not scores:
            return []
        if temperature <= 0:
            temperature = 1e-6
        m = max(scores)
        exps = [math.exp((s - m) / temperature) for s in scores]
        Z = sum(exps)
        if Z <= 0:
            n = len(scores)
            return [1.0 / n] * n
        probs = [e / Z for e in exps]
        if self.ROLLOUT_UNIFORM_MIX > 0:
            n = len(probs)
            mix = self.ROLLOUT_UNIFORM_MIX
            probs = [(1 - mix) * p + mix * (1.0 / n) for p in probs]
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        return probs

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

    def search(self, iterations=100_000_000):
        for _ in range(iterations):
            node = self.select(self.root)
            expanded = node.expand()
            if expanded and expanded.round_end_reward is not None:
                reward_local = expanded.round_end_reward
                self.backpropagate(expanded, reward_local)
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
        state = node.state.copy()
        while True:
            if state.is_terminal(): break
            moves = state.get_legal_moves()
            if not moves:
                state.end_round()
                continue
            # 選擇動作：依旗標決定是否使用 heuristic
            if self.use_heuristics:
                scores = [self._aggregate_heuristic(state, m) for m in moves]
                weights = self._softmax_weights(scores, self.ROLLOUT_TEMPERATURE)
                move = random.choices(moves, weights=weights, k=1)[0]
            else:
                move = random.choice(moves)
            state.make_move(move)
            if state.get_legal_moves(): # 回合尚未結束才換手
                state.switch_player()

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
