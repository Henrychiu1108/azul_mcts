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
    FIRST_PLAYER_BONUS = 0.9  # 拿首玩家標記的額外偏好（正值）
    INSTANT_PLACE_BONUS = 2  # 新增：完成任一 pattern line 固定加成（不依容量）

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

    # --- 新增：解釋用 heuristic 拆解 (不影響搜尋) ---
    def heuristic_breakdown(self, state: GameState, move):
        details = []
        total = 0.0
        for h in self.heuristics:
            try:
                val = h(state, move)
            except Exception:
                val = 0.0
            details.append((h.__name__, val))
            total += val
        return total, details

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
            total, details = self.heuristic_breakdown(state, mv)
            node = child_map.get(mv)
            visits = node.visits if node else 0
            value = node.value if node else 0.0
            avg = (value / visits) if visits > 0 else 0.0
            out.append({
                'move': mv,
                'total': total,
                'details': details,
                'visits': visits,
                'value': value,
                'avg': avg
            })
        return out

    def __init__(self, root: Node, heuristics=None):
        self.root = root
        self.root_player = root.state.current_player
        self.heuristics = heuristics or [
            self.h_floor_action,
            self.h_first_player_marker,
            self.h_line_progress,      # 合併：即刻完成 + 可行預備
            self.h_ev_row,
            self.h_ev_col,
            # self.h_ev_color,
            self.h_adjacency
        ]

    # ---- Heuristic helpers ----
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
        if move[3] != 'floor':
            return 0.0
        player = state.players[state.current_player]
        source_type, source_index, color = move[0], move[1], move[2]
        if source_type == 'factory':
            tiles_avail = sum(1 for t in state.factories[source_index].tiles if t.color == color)
        else:
            tiles_avail = sum(1 for t in state.center.tiles if t.color == color)
        existing = sum(1 for t in player.floor.tiles if t is not None)
        remaining_slots = max(0, 7 - existing)
        effective_tiles = min(tiles_avail, remaining_slots)
        # 硬編各 slot 嚴重度倍率（第 i 格放置後的放大量化）
        slot_severity = [1.0, 1.0, 1.2, 1.4, 1.6, 2.15, 2.70]
        raw_penalty_sum = 0.0              # 真實 Azul 將扣的合計（不含溢出）
        weighted_penalty_sum = 0.0         # 加權後（用於 heuristic 放大後段風險）
        for k in range(effective_tiles):
            pos = existing + k
            base_p = player.floor.penalties[pos]
            raw_penalty_sum += base_p
            weighted_penalty_sum += base_p * slot_severity[pos]
        overflow_tiles = max(0, tiles_avail - remaining_slots)
        overflow_relief = overflow_tiles * 0.3
        # 只針對 raw 做 clipping（真實規則層面），不混合權重
        clipped_raw = max(raw_penalty_sum, -player.score)
        # 風險因子（低分更寬鬆）
        if player.score < 3:
            risk_factor = 0.1
        elif player.score < 9:
            risk_factor = 0.5
        else:
            risk_factor = 1.0
        adjusted_penalty = clipped_raw * risk_factor
        weight = self.FLOOR_BASE + adjusted_penalty + overflow_relief
        return max(0.03, weight)

    # 新增：首玩家標記獨立 heuristic
    def h_first_player_marker(self, state: GameState, move):
        source_type = move[0]
        if source_type != 'center':
            return 0.0
        if not state.center.has_first_player_marker:
            return 0.0
        # 基礎加成
        base = self.FIRST_PLAYER_BONUS
        # 估計是否接近遊戲結束：任一玩家牆上某列已有 4 格 (row_counts == 4) 代表那列差一格即可觸發終局
        current_player_board = state.players[state.current_player]
        other_player_board = state.players[1 - state.current_player]
        near_rows_cur = sum(1 for v in current_player_board.row_counts if v == 4)
        near_rows_opp = sum(1 for v in other_player_board.row_counts if v == 4)
        danger = near_rows_cur + near_rows_opp
        if danger > 0:
            # 每個臨界行使先手價值遞減，最多大幅壓低；線性轉衰減係數
            decay = 1.0 / (1.0 + 0.6 * danger)  # danger=1 -> ~0.625, 2->~0.454, 3->~0.357
            base *= decay
        # 若幾乎確定終局（任一玩家已有兩條以上 4 格列）再額外再衰減一點
        if danger >= 2:
            base *= 0.85
        return max(0.05, base)

    def h_line_progress(self, state: GameState, move):
        """合併即刻完成 (原 h_instant_place) 與可望於本輪完成的預備 (原 h_prep_line)。
        邏輯：
          1. 不是 pattern line → 0
          2. 若本手補滿該行 → INSTANT_PLACE_BONUS
          3. 否則計算 after 進度；若剩餘需瓷磚數 > 3 → 放棄
          4. 估算剩餘同色池（不含本次取走）；若不足以完成 → 0
          5. 根據剩餘需求 (1/2/3) 給 feasibility；再乘進度比例與供給比供應 (supply_factor)
          6. 最終：score = PREP_FACTOR * progress_ratio * feasibility * supply_factor * capacity_scale
        目的：在不立即完成時仍鼓勵合理投資有望完成的 pattern line。"""
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        line = player.pattern_lines.lines[dest_row]
        capacity = dest_row + 1
        current = len(line)
        source_type, source_index = move[0], move[1]
        if source_type == 'factory':
            tiles_avail = sum(1 for t in state.factories[source_index].tiles if t.color == color)
        else:
            tiles_avail = sum(1 for t in state.center.tiles if t.color == color)
        placeable = min(tiles_avail, capacity - current)
        after = current + placeable
        # Case 1: 立即完成
        if after == capacity:
            return self.INSTANT_PLACE_BONUS
        remaining_needed = capacity - after
        if remaining_needed > 3:
            return 0.0
        # 估算盤面尚存該色（扣除此次取走）
        total_color = 0
        for f in state.factories:
            total_color += sum(1 for t in f.tiles if t.color == color)
        total_color += sum(1 for t in state.center.tiles if t.color == color)
        remaining_color_pool = max(0, total_color - tiles_avail)
        if remaining_color_pool < remaining_needed:
            return 0.0
        # 進度比例
        progress_ratio = after / capacity
        # 剩餘需求可行性
        if remaining_needed == 1:
            feasibility = 1.0
        elif remaining_needed == 2:
            feasibility = 0.65
        else:  # ==3
            feasibility = 0.35
        # 供給比例（越多剩餘越安心，封頂 1.3 以免膨脹）
        supply_factor = min(1.3, remaining_color_pool / remaining_needed)
        # 高容量行前期投入微幅加成
        capacity_scale = 0.85 + 0.15 * (capacity / 5.0)
        score = self.PREP_FACTOR * progress_ratio * feasibility * supply_factor * capacity_scale
        return max(score, 0.02) if score > 0 else 0.0

    # 2. 期望值：行完成（移除立即放置判斷）
    def h_ev_row(self, state: GameState, move):
        """行完成進展（Row EV）：非線性後段加權 + gap==1 kicker。"""
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.row_counts[dest_row], self.ROW_BONUS, 2.2, 0.55)

    # 3. 期望值：列完成（移除立即放置判斷）
    def h_ev_col(self, state: GameState, move):
        """列完成進展（Column EV）：較尖銳指數 + gap==1 kicker。"""
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.col_counts[column], self.COL_BONUS, 2.4, 0.75)

    # 4. 期望值：顏色完成（移除立即放置判斷）
    def h_ev_color(self, state: GameState, move):
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        new_color_count = player.color_counts[color] + 1
        gap = 5 - new_color_count
        prob = self.COLOR_COMPLETE_PROB.get(gap, 0.0)
        return prob * self.COLOR_BONUS

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
        total = 0.0
        for h in self.heuristics:
            try:
                total += h(state, move)
            except Exception:
                continue
        return max(total, 0.001)

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

    def search(self, iterations=10_000_000):
        for _ in range(iterations):
            node = self.select(self.root)
            expanded = node.expand()
            if expanded and expanded.round_end_reward is not None:
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
        state = node.state.copy()
        while True:
            moves = state.get_legal_moves()
            if not moves:
                break
            scores = [self._aggregate_heuristic(state, m) for m in moves]
            move = random.choices(moves, weights=scores, k=1)[0]
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
