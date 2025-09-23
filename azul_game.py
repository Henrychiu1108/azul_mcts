import random
from typing import List, Dict, Tuple
from enum import Enum
import numpy as np
import copy

class Color(Enum):
    BLUE = "blue"
    RED = "red"
    BLACK = "black"
    GREEN = "green"
    YELLOW = "yellow"

class Tile:
    def __init__(self, color: Color):
        self.color = color

class Factory:
    def __init__(self, tiles: List[Tile]):
        self.tiles = tiles

class Board:
    def __init__(self):
        self.pattern = np.array([
            [Color.YELLOW, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK],
            [Color.BLACK, Color.YELLOW, Color.BLUE, Color.GREEN, Color.RED],
            [Color.RED, Color.BLACK, Color.YELLOW, Color.BLUE, Color.GREEN],
            [Color.GREEN, Color.RED, Color.BLACK, Color.YELLOW, Color.BLUE],
            [Color.BLUE, Color.GREEN, Color.RED, Color.BLACK, Color.YELLOW]
        ], dtype=object)  # 5x5 pattern for accepted colors
        self.occupancy = np.zeros((5, 5), dtype=int)  # 5x5 occupancy: 0 empty, 1 occupied
        self.color_positions = {color: [] for color in Color}
        for i in range(5):
            for j in range(5):
                self.color_positions[self.pattern[i, j]].append((i, j))

class PlayerBoard:
    def __init__(self):
        self.board = Board()  # The player's board with pattern and occupancy
        self.pattern_lines = PatternLine(self.board)  # Pattern lines for staging tiles
        self.floor = Floor()  # The player's floor for penalty tiles
        self.score = 0

class Bag:
    def __init__(self):
        self.tiles = [Tile(color) for color in Color for _ in range(20)]  # 20 of each color
        random.shuffle(self.tiles)

    def draw(self):
        """Draw a tile from the bag."""
        if self.tiles:
            return self.tiles.pop()
        return None

    def is_empty(self):
        return len(self.tiles) == 0

class Discard:
    def __init__(self):
        self.tiles = []

    def add(self, tile: Tile):
        """Add a tile to the discard."""
        self.tiles.append(tile)

    def is_empty(self):
        return len(self.tiles) == 0

class Center:
    def __init__(self):
        self.tiles = []
        self.has_first_player_marker = True

class Floor:
    def __init__(self):
        self.penalties = [-1, -1, -2, -2, -2, -3, -3]  # Penalty scores for each position
        self.tiles = [None] * 7  # 7 positions, each can hold a tile or None

class PatternLine:
    def __init__(self, board):
        self.lines = [[] for _ in range(5)]  # 5 lines for staging tiles
        self.board = board  # Reference to the player's board for restrictions

    def can_place(self, color: Color, line_index: int) -> bool:
        """Check if a tile of the given color can be placed in the specified line."""
        if not (0 <= line_index < 5):
            return False
        line = self.lines[line_index]
        # 使用 is_full 簡化容量判斷
        if self.is_full(line_index):
            return False
        # Check color consistency (all tiles in line must be same color)
        if line and line[0].color != color:
            return False
        # Check wall restriction (cannot place color if already in wall row)
        for j in range(5):
            if self.board.occupancy[line_index, j] == 1 and self.board.pattern[line_index, j] == color:
                return False
        return True

    def place_tile(self, tile: Tile, line_index: int) -> bool:
        """Place a tile in the specified line if possible."""
        if self.can_place(tile.color, line_index):
            self.lines[line_index].append(tile)
            return True
        return False

    def is_full(self, line_index: int) -> bool:
        """Check if the line is full."""
        return len(self.lines[line_index]) == line_index + 1
    
class GameState:
    def __init__(self):
        self.factories = [Factory([]) for _ in range(5)]  # 5 factories
        self.center = Center()  # Center instance
        self.players = [PlayerBoard(), PlayerBoard()]
        self.current_player = 0
        self.bag = Bag()  # Bag instance
        self.discard = Discard()  # Discard instance
        # 以玩家 0 為起始持有人（首回合即為起始玩家）
        self.first_player_marker_holder = 0

    # 共用：計算把 (row,col) 放入後的相鄰得分（符合官方 Azul 規則）
    @staticmethod
    def _adjacency_score(occ: np.ndarray, row: int, col: int) -> int:
        """Official Azul scoring for a newly placed tile:
        - Compute horizontal group length (including the tile) if any adjacent horizontally.
        - Compute vertical group length (including the tile) if any adjacent vertically.
        - If both groups size > 1, score = horizontal_length + vertical_length (tile counted twice).
        - Else score = max(horizontal_length, vertical_length)."""
        # Horizontal length
        h_len = 1
        c = col - 1
        while c >= 0 and occ[row, c] == 1:
            h_len += 1
            c -= 1
        c = col + 1
        while c < 5 and occ[row, c] == 1:
            h_len += 1
            c += 1
        # Vertical length
        v_len = 1
        r = row - 1
        while r >= 0 and occ[r, col] == 1:
            v_len += 1
            r -= 1
        r = row + 1
        while r < 5 and occ[r, col] == 1:
            v_len += 1
            r += 1
        if h_len > 1 and v_len > 1:
            return h_len + v_len  # tile counted twice
        return max(h_len, v_len)

    @staticmethod
    def _floor_penalty_value(floor) -> int:
        """純計算地板懲罰分（負值），不更動任何狀態。"""
        total = 0
        for i, tile in enumerate(floor.tiles):
            if tile is not None:
                total += floor.penalties[i]
        return total

    def evaluate_floor_penalties(self) -> List[int]:
        """回傳目前每位玩家若立即結算的地板懲罰總和（負值）。"""
        return [self._floor_penalty_value(player.floor) for player in self.players]

    def _compute_full_line_placements(self, player) -> List[Tuple[int, int, int, Color]]:
        """回傳本回合結算時，該玩家所有『已滿 pattern line』對應的放置資訊列表。
        每項: (row, col, gained_score, color)
        使用行索引 0..4 的順序，並模擬逐行放置（影響後續相鄰得分）。不改動真實 occupancy。"""
        placements = []
        temp_occ = player.board.occupancy.copy()
        for line_idx in range(5):
            line = player.pattern_lines.lines[line_idx]
            if len(line) == line_idx + 1:  # 已滿
                tile = line[0]
                color = tile.color
                for col in range(5):
                    if player.board.pattern[line_idx, col] == color:
                        temp_occ[line_idx, col] = 1
                        gained = self._adjacency_score(temp_occ, line_idx, col)
                        placements.append((line_idx, col, gained, color))
                        break
        return placements

    def is_terminal(self) -> bool:
        """Check if the game is over (a player has completed a row on the wall)."""
        for player in self.players:
            # Check rows
            for i in range(5):
                if all(player.board.occupancy[i, j] == 1 for j in range(5)):
                    return True
        return False

    def switch_player(self):
        """切換目前玩家（單純輪替，不處理回合結束邏輯）。"""
        self.current_player = 1 - self.current_player

    def get_legal_moves(self) -> List[Tuple[str, int, Color, str, int]]:
        """Return legal moves. 若回傳空列表，表示本回合已無可行動（工廠與中心皆空）。"""
        moves: List[Tuple[str, int, Color, str, int]] = []
        player = self.players[self.current_player]
        
        def get_colors_from_source(source_type, source_index):
            if source_type == 'factory':
                factory = self.factories[source_index]
                return set(tile.color for tile in factory.tiles)
            if source_type == 'center':
                return set(tile.color for tile in self.center.tiles)
            return set()
        
        # Factories
        for factory_idx in range(5):
            if self.factories[factory_idx].tiles:
                available_colors = get_colors_from_source('factory', factory_idx)
                for color in available_colors:
                    for line_idx in range(5):
                        if player.pattern_lines.can_place(color, line_idx):
                            moves.append(('factory', factory_idx, color, 'pattern_line', line_idx))
                    moves.append(('factory', factory_idx, color, 'floor', None))
        
        # Center
        if self.center.tiles:
            available_colors = get_colors_from_source('center', None)
            for color in available_colors:
                for line_idx in range(5):
                    if player.pattern_lines.can_place(color, line_idx):
                        moves.append(('center', None, color, 'pattern_line', line_idx))
                moves.append(('center', None, color, 'floor', None))
        
        return moves

    def make_move(self, move: Tuple[str, int, Color, str, int]):
        """Apply a move to the game state."""
        source_type, source_index, color, destination_type, destination_index = move
        player = self.players[self.current_player]
        
        taken_tiles = []
        if source_type == 'factory':
            factory = self.factories[source_index]
            taken_tiles = [tile for tile in factory.tiles if tile.color == color]
            factory.tiles = [tile for tile in factory.tiles if tile.color != color]
            for tile in factory.tiles:
                self.center.tiles.append(tile)
            factory.tiles = []
        elif source_type == 'center':
            taken_tiles = [tile for tile in self.center.tiles if tile.color == color]
            self.center.tiles = [tile for tile in self.center.tiles if tile.color != color]
            if self.center.has_first_player_marker:
                # 直接放置標記到地板，不加入 taken_tiles
                if player.floor.tiles[6] is not None:
                    self.discard.add(player.floor.tiles[6])
                for i in range(6, 0, -1):
                    player.floor.tiles[i] = player.floor.tiles[i-1]
                player.floor.tiles[0] = FirstPlayerMarker()
                self.center.has_first_player_marker = False
                self.first_player_marker_holder = self.current_player
        
        def _add_to_floor(tile_obj):
            for i in range(7):
                if player.floor.tiles[i] is None:
                    player.floor.tiles[i] = tile_obj
                    return
            self.discard.add(tile_obj)
        
        if destination_type == 'pattern_line':
            # 使用 is_full 檢查是否已滿；逐一放入，多餘進地板
            for tile in taken_tiles:
                if player.pattern_lines.is_full(destination_index):
                    _add_to_floor(tile)
                else:
                    player.pattern_lines.lines[destination_index].append(tile)
        elif destination_type == 'floor':
            for tile in taken_tiles:
                _add_to_floor(tile)
        
    def get_winner(self):
        """Return the winner (player index) or None if tie, and bonuses."""
        
        bonuses = []
        # Calculate end-game bonuses
        for player_idx, player in enumerate(self.players):
            row_bonuses = 0
            col_bonuses = 0
            color_bonuses = 0
            
            # Bonus for complete rows: 2 points each
            for i in range(5):
                if all(player.board.occupancy[i, j] == 1 for j in range(5)):
                    player.score += 2
                    row_bonuses += 1
            
            # Bonus for complete columns: 7 points each
            for j in range(5):
                if all(player.board.occupancy[i, j] == 1 for i in range(5)):
                    player.score += 7
                    col_bonuses += 1
            
            # Bonus for complete color groups: 10 points each
            for color in Color:
                positions = player.board.color_positions[color]
                if all(player.board.occupancy[pos[0], pos[1]] == 1 for pos in positions):
                    player.score += 10
                    color_bonuses += 1
            
            bonuses.append((row_bonuses, col_bonuses, color_bonuses))
            # Print bonus details for this player
            total_bonus = row_bonuses * 2 + col_bonuses * 7 + color_bonuses * 10
            print(f"Player {player_idx} end-game bonuses: rows={row_bonuses} (+{row_bonuses*2}), cols={col_bonuses} (+{col_bonuses*7}), colors={color_bonuses} (+{color_bonuses*10}) => total +{total_bonus}")
        
        scores = [player.score for player in self.players]
        max_score = max(scores)
        winners = [i for i, score in enumerate(scores) if score == max_score]
        if len(winners) == 1:
            return winners[0]
        return None  # Tie

    def refill_factories(self):
        """Refill factories with 4 tiles each from bag. 若需且可，將棄牌洗回。若 bag 與 discard 皆空則拋出錯誤。"""
        for factory in self.factories:
            factory.tiles = []
            for _ in range(4):
                if self.bag.is_empty():
                    self.bag.tiles.extend(self.discard.tiles)
                    self.discard.tiles = []
                    random.shuffle(self.bag.tiles)
                tile = self.bag.draw()
                if tile is None:
                    raise RuntimeError("refill_factories: unexpected None tile after refill logic")
                factory.tiles.append(tile)

    def end_round(self):
        """正式回合結算：更新牆、丟棄、地板懲罰、補工廠。"""
        for player_idx, player in enumerate(self.players):
            placements = self._compute_full_line_placements(player)
            # 實際放置並加分（這裡再放一次到真實 occupancy）
            for row, col, gained, color in placements:
                player.board.occupancy[row, col] = 1
                player.score += gained
                # 丟棄多餘 tiles 並清空該 pattern line
                line = player.pattern_lines.lines[row]
                for extra in line[1:]:
                    self.discard.add(extra)
                player.pattern_lines.lines[row] = []
            # 地板懲罰加總
            player.score += self._floor_penalty_value(player.floor)
            # 處理地板實際 tiles 移動（標記 / 丟棄）
            for tile in player.floor.tiles:
                if tile is None:
                    continue
                if isinstance(tile, FirstPlayerMarker):
                    self.center.has_first_player_marker = True
                else:
                    self.discard.add(tile)
            player.score = max(0, player.score)
            player.floor.tiles = [None] * 7
        self.current_player = self.first_player_marker_holder
        self.refill_factories()
        self.center.tiles = []
        self.center.has_first_player_marker = True

    def copy(self):
        """Return a deep copy of the game state."""
        return copy.deepcopy(self)



class FirstPlayerMarker:
    pass

