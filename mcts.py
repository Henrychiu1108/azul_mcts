import math
import random
from azul_game import GameState  # 移除未使用 FirstPlayerMarker

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
    # 機率表：gap = 尚需幾格(放進後) 才能完成 (row/col/color)。列與顏色較難完成 → 機率較低。
    COLOR_COMPLETE_PROB = {0:1.0, 1:0.45, 2:0.22, 3:0.09, 4:0.02}
    ROW_BONUS   = 5
    COL_BONUS   = 5
    COLOR_BONUS = 10
    # 折減：若本手不會立即完成該 pattern line（不會立刻放上牆），視為只是“預備”，給預備係數 < 1
    PREP_FACTOR = 0.3
    # 新增：地板動態評估參數
    FLOOR_BASE = 1.0          # 基礎基準
    FIRST_PLAYER_BONUS = 1.2  # 拿首玩家標記的額外偏好（正值）
    INSTANT_PLACE_BONUS = 1.3  # 完成任一 pattern line 固定加成（不依容量）
    # Softmax rollout 參數
    ROLLOUT_TEMPERATURE = 1.2   # 溫度 (越大越平均)
    ROLLOUT_UNIFORM_MIX = 0.15  # 與均勻分布混合比例，避免機率過度趨零
    # --- 恢復：自適應權重參數 ---
    WEIGHT_LEARNING_RATE = 0.002
    WEIGHT_L2_DECAY = 0.0005
    WEIGHT_MIN = 0.05
    WEIGHT_MAX = 5.0
    WEIGHT_TARGET_MEAN = 1.0
    WEIGHT_CLIP_NORM = 6.0

    # --- 新增：行 / 列 進度共用 helper ---
    def _h_ev_progress(self, current_count: int, bonus: float, exponent: float, kicker_gap1: float) -> float:
        """抽取行/列進度評估共用邏輯：
        current_count: 目前該行或該列已放牆數
        bonus: 基礎獎勵（ROW_BONUS 或 COL_BONUS）
        exponent: 非線性 shaping 指數（行 2.2 / 列 2.4）
        kicker_gap1: gap==1 時額外加成（行 0.55 / 列 0.75）"""
        new_count = min(5, current_count + 1)  # 假設此顆最終會放上牆
        gap = 5 - new_count
        progress_ratio = new_count / 5.0
        shaped = progress_ratio ** exponent
        return bonus * shaped + (kicker_gap1 if gap == 1 else 0.0)

    # --- 新增：地板懲罰共用 helper ---
    def _floor_penalty_heuristic(self, player, tiles_to_floor: int):
        """給定本手將實際落到地板的瓷磚數，估算地板懲罰特徵值。
        tiles_to_floor 可為 0（則回 0）。包含：
          - 直接 floor 動作所有取得的同色數
          - pattern line 溢出部分
        """
        if tiles_to_floor <= 0:
            return 0.0
        existing = sum(1 for t in player.floor.tiles if t is not None)
        remaining_slots = max(0, 7 - existing)
        effective_tiles = min(tiles_to_floor, remaining_slots)
        raw_penalty_sum = 0.0
        for k in range(effective_tiles):
            pos = existing + k
            base_p = player.floor.penalties[pos]
            raw_penalty_sum += base_p
        overflow_tiles = max(0, tiles_to_floor - remaining_slots)
        overflow_relief = overflow_tiles * 0.3
        clipped_raw = max(raw_penalty_sum, -player.score)
        if player.score < 3:
            risk_factor = 0.01
        elif player.score < 9:
            risk_factor = 0.05
        else:
            risk_factor = 1.0
        adjusted_penalty = clipped_raw * risk_factor
        weight = self.FLOOR_BASE + adjusted_penalty + overflow_relief
        return max(0.05, weight)

    # --- 還原：解釋 + 加權 ---
    def heuristic_breakdown(self, state: GameState, move):
        """回傳 (加權總分, [(名稱, 原值), ...], raw_feature_list)。
        raw_feature_list 供權重學習；加權總分 = Σ w_i * raw_i。"""
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
        """回傳目前 state 所有合法手之 heuristic 拆解與（若已擴展）子節點統計。"""
        moves = state.get_legal_moves()
        out = []
        # 建立 move -> child 快取 (root only)
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
                'raw': raw,             # 新增：原始特徵
                'visits': visits,
                'value': value,
                'avg': avg
            })
        # --- 新增：計算 heuristic / Q 排名與相關指標 ---
        if not out:
            return out
        # 排序（heuristic total 高到低）
        sorted_h = sorted(out, key=lambda x: x['total'], reverse=True)
        # 排序（Q = avg，高到低，未訪問視為極小）
        sorted_q = sorted(out, key=lambda x: (x['avg'] if x['visits']>0 else -1e9), reverse=True)
        h_rank_map = {id(m): i for i, m in enumerate(sorted_h)}
        q_rank_map = {id(m): i for i, m in enumerate(sorted_q)}
        # Spearman：僅計算有 visits 的節點（避免全部 -1e9 扰動）；若不足兩個則 None
        visited = [m for m in out if m['visits']>0]
        spearman = ''
        if len(visited) >= 2:
            # 對共同集合（此處即 visited）用 rank
            d2 = 0
            n = len(visited)
            for m in visited:
                d = h_rank_map[id(m)] - q_rank_map[id(m)]
                d2 += d*d
            spearman_val = 1 - 6*d2/(n*(n*n-1)) if n > 1 else 0.0
            spearman = f"{spearman_val:.4f}"
        # 確定 top 是否一致
        top_match = ''
        if sorted_h and sorted_q:
            top_match = int(sorted_h[0]['move'] == sorted_q[0]['move'])
        num_moves = len(out)
        # 寫回
        for m in out:
            m['heuristic_rank'] = h_rank_map[id(m)]
            m['q_rank'] = q_rank_map[id(m)]
            m['spearman'] = spearman
            m['top_match'] = top_match
            m['num_moves'] = num_moves
        return out

    def __init__(self, root: Node, heuristics=None):
        self.root = root
        self.root_player = root.state.current_player
        self.heuristics = heuristics or [
            self.h_floor_action,
            self.h_first_player_marker,
            self.h_line_progress,
            self.h_ev_row,
            self.h_ev_col,
            self.h_adjacency
        ]
        # 初始化可學習權重
        self.weights = [1.0] * len(self.heuristics)

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

    # 新增：專責地板行為評估（加入：分數底線與容量飽和）
    def h_floor_action(self, state: GameState, move):
        """統一：
        - 直接放地板: 以全部同色取得數量計算。
        - pattern_line 若有溢出到地板: 只以溢出部分計算。
        其餘不影響地板者回 0。"""
        source_type, source_index, color, dest_type = move[0], move[1], move[2], move[3]
        player = state.players[state.current_player]
        # 取得本次取走的該色總數
        if source_type == 'factory':
            tiles_avail = sum(1 for t in state.factories[source_index].tiles if t.color == color)
        else:  # center
            tiles_avail = sum(1 for t in state.center.tiles if t.color == color)
        if dest_type == 'floor':
            return self._floor_penalty_heuristic(player, tiles_avail)
        if dest_type == 'pattern_line':
            dest_row = move[4]
            line = player.pattern_lines.lines[dest_row]
            capacity = dest_row + 1
            current = len(line)
            placeable = min(tiles_avail, capacity - current)
            overflow_to_floor = max(0, tiles_avail - placeable)
            return self._floor_penalty_heuristic(player, overflow_to_floor)
        return 0.0

    # 新增：首玩家標記獨立 heuristic
    def h_first_player_marker(self, state: GameState, move):
        source_type = move[0]
        if source_type != 'center':
            return 0.0
        if not state.center.has_first_player_marker:
            return 0.0
        base = self.FIRST_PLAYER_BONUS
        current_player_board = state.players[state.current_player]
        other_player_board = state.players[1 - state.current_player]
        # danger: 只要任一玩家有任一列已達 4 (差一即可終局)
        danger_flag = any(v == 4 for v in current_player_board.row_counts) or \
                       any(v == 4 for v in other_player_board.row_counts)
        if danger_flag:
            # 危險態勢下先手重要性上升（放大而非衰減）
            base *= 1.2  # 可日後調整或改為參數
        return max(0.05, base)

    def h_line_progress(self, state: GameState, move):
        """最終重構版：單一 ratio = (場上該色總數) / capacity。
        規則：
          ratio < 1: 永遠無法完成，給負分 → penalty = SCARCITY_MISSING_PENALTY * (1 - ratio)
          ratio ~= 1: 剛好足夠，給峰值 SCARCITY_EXACT_BONUS
          ratio > 1: 顏色充裕，隨 ratio 遞增做指數衰減：bonus = EXACT * exp(-DECAY * (ratio-1))，但不低於 SCARCITY_ABUNDANCE_MIN
        不再考慮 placeable / progress / remaining_needed；交由學習權重決定重要性。
        """
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        line = player.pattern_lines.lines[dest_row]
        capacity = dest_row + 1
        current = len(line)
        # 統計場上該色總數（含工廠 + 中央），使用『取前』狀態
        total_color = 0
        for f in state.factories:
            total_color += sum(1 for t in f.tiles if t.color == color)
        total_color += sum(1 for t in state.center.tiles if t.color == color)
        ratio = total_color / capacity
        # 判斷區間
        eps = 1e-6
        if ratio < 1 - eps:
            # 不足：無法完成 → 負向懲罰（缺多少乘多少）
            score = self.SCARCITY_MISSING_PENALTY * (1 - ratio)
        elif abs(ratio - 1) <= eps:
            score = self.SCARCITY_EXACT_BONUS
        else:
            over = ratio - 1.0
            score = self.SCARCITY_EXACT_BONUS * math.exp(-self.SCARCITY_ABUNDANCE_DECAY * over)
            if score < self.SCARCITY_ABUNDANCE_MIN:
                score = self.SCARCITY_ABUNDANCE_MIN
        return score

    # 新增：地板懲罰（移除立即放置判斷）
    def h_ev_row(self, state: GameState, move):
        """行完成進展（Row EV）：非線性後段加權 + gap==1 kicker。"""
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.row_counts[dest_row], self.ROW_BONUS, 2.4, 0.55)

    # 3. 期望值：列完成（移除立即放置判斷）
    def h_ev_col(self, state: GameState, move):
        """列完成進展（Column EV）：較尖銳指數 + gap==1 kicker。"""
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.col_counts[column], self.COL_BONUS, 2.2, 0.75)


    # 5. 相鄰潛力（移除立即放置判斷，改為預估：只在實際牆上鄰居存在時仍給分）
    def h_adjacency(self, state: GameState, move):
        is_pattern, player, dest_row, color, column, occ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        # 仍沿用現有牆占據資訊（若尚未放置，此時 occ 該位置為 0，只計算既有鄰居帶來的潛在鏈接）
        h_len = 1
        c = column - 1
        while c >= 0 and occ[dest_row, c] == 1:
            h_len += 1; c -= 1
        c = column + 1
        while c < 5 and occ[dest_row, c] == 1:
            h_len += 1; c += 1
        v_len = 1
        r = dest_row - 1
        while r >= 0 and occ[r, column] == 1:
            v_len += 1; r -= 1
        r = dest_row + 1
        while r < 5 and occ[r, column] == 1:
            v_len += 1; r += 1
        adjacency = (h_len + v_len) if (h_len > 1 and v_len > 1) else max(h_len, v_len)
        return adjacency

    def _aggregate_heuristic(self, state: GameState, move):
        """允許加總為負（表示明顯不佳），交由 softmax 平移處理。"""
        total = 0.0
        for w, h in zip(self.weights, self.heuristics):
            try:
                v = h(state, move)
            except Exception:
                v = 0.0
            total += w * v
        return total

    # --- 新增：softmax 權重計算，避免壓死低分手 ---
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
        # 與均勻分佈混合，保底探索
        if self.ROLLOUT_UNIFORM_MIX > 0:
            n = len(probs)
            mix = self.ROLLOUT_UNIFORM_MIX
            probs = [(1 - mix) * p + mix * (1.0 / n) for p in probs]
        # 再次正規化（數值安全）
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        return probs

    # --- 恢復：回合結束（來源耗盡）暫時計分差評估 ---
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

    def search(self, iterations=50_000_000):
        for _ in range(iterations):
            node = self.select(self.root)
            expanded = node.expand()
            if expanded and expanded.round_end_reward is not None:
                reward_local = expanded.round_end_reward
                self.backpropagate(expanded, reward_local)
                self._maybe_update_weights(expanded, reward_local)
                continue
            node_to_sim = expanded if expanded else node
            reward = self.simulate(node_to_sim)
            self.backpropagate(node_to_sim, reward)
            self._maybe_update_weights(node_to_sim, reward)

    def _maybe_update_weights(self, leaf_node: Node, reward):
        # 找 root child
        curr = leaf_node
        if curr is None:
            return
        while curr.parent is not None and curr.parent != self.root:
            curr = curr.parent
        if curr.parent != self.root:
            return
        move = curr.move
        if move is None:
            return
        raw_vals = []
        for h in self.heuristics:
            try:
                raw_vals.append(h(self.root.state, move))
            except Exception:
                raw_vals.append(0.0)
        predicted = sum(w * x for w, x in zip(self.weights, raw_vals))
        error = reward - predicted
        if error == 0:
            return
        lr = self.WEIGHT_LEARNING_RATE
        l2 = self.WEIGHT_L2_DECAY
        for i, (w, x) in enumerate(zip(self.weights, raw_vals)):
            grad = error * x - l2 * w
            new_w = w + lr * grad
            if new_w < self.WEIGHT_MIN:
                new_w = self.WEIGHT_MIN
            elif new_w > self.WEIGHT_MAX:
                new_w = self.WEIGHT_MAX
            self.weights[i] = new_w
        mean_w = sum(self.weights)/len(self.weights)
        if mean_w > 0:
            scale = self.WEIGHT_TARGET_MEAN / mean_w
            self.weights = [max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, w*scale)) for w in self.weights]
        if self.WEIGHT_CLIP_NORM is not None:
            norm = math.sqrt(sum(w*w for w in self.weights))
            if norm > self.WEIGHT_CLIP_NORM:
                ratio = self.WEIGHT_CLIP_NORM / norm
                self.weights = [w*ratio for w in self.weights]

    def select(self, node: Node):
        while node.is_fully_expanded():
            if not node.state.get_legal_moves():
                break
            node = node.best_child()
        return node

    def simulate(self, node: Node):
        state = node.state.copy()
        while True:
            moves = state.get_legal_moves()
            if not moves:
                break
            scores = [self._aggregate_heuristic(state, m) for m in moves]
            weights = self._softmax_weights(scores, self.ROLLOUT_TEMPERATURE)
            move = random.choices(moves, weights=weights, k=1)[0]
            state.make_move(move)
            if state.get_legal_moves():
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
