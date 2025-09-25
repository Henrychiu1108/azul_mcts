import math
from azul_game import GameState

class HeuristicMixin:
    # Heuristic-related constants (trimmed to only those still in use)
    ROW_BONUS   = 5
    COL_BONUS   = 5
    FIRST_PLAYER_BONUS = 1.2
    ADJ_NO_NEIGHBOR_BASE = 0.35

    # Shared progress helper
    def _h_ev_progress(self, current_count: int, bonus: float, exponent: float, kicker_gap1: float) -> float:
        new_count = current_count + 1
        gap = 5 - new_count
        progress_ratio = (current_count + 1) / 5.0
        shaped = progress_ratio ** exponent
        return bonus * shaped + (kicker_gap1 if gap == 1 else 0.0)

    # Shared floor penalty helper
    def _floor_penalty_heuristic(self, player, tiles_to_floor: int):
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
        weight =  adjusted_penalty + overflow_relief
        return weight

    # Heuristic: floor action / overflow
    def h_floor_action(self, state: GameState, move):
        source_type, source_index, color, dest_type = move[0], move[1], move[2], move[3]
        player = state.players[state.current_player]
        if source_type == 'factory':
            tiles_avail = sum(1 for t in state.factories[source_index].tiles if t.color == color)
        else:
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

    # Heuristic: first player marker
    def h_first_player_marker(self, state: GameState, move):
        source_type = move[0]
        if source_type != 'center':
            return 0.0
        if not state.center.has_first_player_marker:
            return 0.0
        base = self.FIRST_PLAYER_BONUS
        current_player_board = state.players[state.current_player]
        other_player_board = state.players[1 - state.current_player]
        danger_flag = any(v == 4 for v in current_player_board.row_counts) or \
                       any(v == 4 for v in other_player_board.row_counts)
        if danger_flag:
            base *= 1.2
        return max(0.05, base)

    # Heuristic: line progress (refactored with ratio indicators)
    def h_line_progress(self, state: GameState, move):
        """Pattern line 進度權重 (新版本)
        指標定義:
          taken  = 此次從來源拿到的該色數量 (全部拿走, 含可能溢出)
          slot   = 目標 pattern line 剩餘可放入的格數 (remaining slots)
          total  = 全場(所有工廠 + 中央)目前該色總數 (拿之前)
          sources= 目前仍含該色的來源數 (工廠(>=1)計1, 中央(>=1)再加1)
        權重考量:
          1. taken/slot < 1 視為不足(偏弱), > 1 視為超量搶 (偏強)
          2. total/slot 越大代表供應充足 → 可安心填這行 (加分)
          3. sources/slot 越大代表分散, 不易被單一行動抽乾 → 風險低 (加分)
        合成方式: 取三個 component 之乘積 (皆以>0為基準, ~1 為中性)。
        """
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        capacity = dest_row + 1
        line = player.pattern_lines.lines[dest_row]
        current = len(line)
        slot = capacity - current
        if slot <= 0:
            return 0.0

        # taken
        source_type, source_index = move[0], move[1]
        if source_type == 'factory':
            taken = sum(1 for t in state.factories[source_index].tiles if t.color == color)
        else:
            taken = sum(1 for t in state.center.tiles if t.color == color)
        if taken <= 0:
            return 0.0

        # total & sources
        total = 0
        sources = 0
        for f in state.factories:
            c = sum(1 for t in f.tiles if t.color == color)
            if c > 0:
                total += c
                sources += 1
        center_c = sum(1 for t in state.center.tiles if t.color == color)
        if center_c > 0:
            total += center_c
            sources += 1
        if total == 0:
            return 0.0

        # 1. taken/slot ratio component
        r_taken = taken / slot
        if r_taken < 1:
            # 次方壓低 (<1 仍保持正, 代表不理想)
            comp_taken = r_taken ** 1.7  # <1 → 降
        else:
            # 超量：log 緩增避免爆炸
            comp_taken = 1.0 + math.log(r_taken)  # r_taken=1 →1

        # 2. total/slot ratio component (方案 B: 完成機率 * 充裕度緩增，僅對高索引行放大)
        r_total = total / slot
        line_importance = (dest_row + 1) / 5.0  # 0.2 ~ 1.0
        completion_prob = min(1.0, r_total)
        if r_total <= 1:
            abundance_gain = 0.7
        else:
            abundance_gain = 1 + 0.5 * (1 - math.exp(-0.8 * (r_total - 1)))  # 漸進上限 1.5
        comp_total_raw = (0.15 + 0.85 * completion_prob) * abundance_gain
        comp_total = 1 + (comp_total_raw - 1) * line_importance  # 低行淡化，高行保留增益

        # 3. sources/slot ratio component (來源分散對高索引行更重要，低索引行淡化)
        r_sources = sources / slot
        base_sources = (r_sources ** 0.5)
        comp_sources = 1 + (base_sources - 1) * line_importance

        # 合成 (基準 ~1, 低於1 下壓, 高於1 提升)
        weight = comp_taken * comp_total * comp_sources
        return weight

    # Heuristic: row progress expected value
    def h_ev_row(self, state: GameState, move):
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.row_counts[dest_row], self.ROW_BONUS, 2.4, 0.55)

    # Heuristic: column progress expected value
    def h_ev_col(self, state: GameState, move):
        is_pattern, player, dest_row, color, column, _ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
        return self._h_ev_progress(player.col_counts[column], self.COL_BONUS, 2.2, 0.75)

    # Heuristic: adjacency potential
    def h_adjacency(self, state: GameState, move):
        is_pattern, player, dest_row, color, column, occ = self._extract_move_context(state, move)
        if not is_pattern:
            return 0.0
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
        h_adj = h_len - 1
        v_adj = v_len - 1
        if h_adj == 0 and v_adj == 0:
            return self.ADJ_NO_NEIGHBOR_BASE
        if h_adj > 0 and v_adj > 0:
            return h_adj + v_adj
        return max(h_adj, v_adj)
