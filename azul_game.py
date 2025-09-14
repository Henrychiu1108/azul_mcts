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

class GameState:
    def __init__(self):
        self.factories = [Factory([]) for _ in range(5)]  # 5 factories
        self.center = Center()  # Center instance
        self.players = [PlayerBoard(), PlayerBoard()]
        self.current_player = 0
        self.bag = Bag()  # Bag instance
        self.discard = Discard()  # Discard instance
        self.first_player_marker_holder = None

    def is_terminal(self) -> bool:
        """Check if the game is over (a player has completed a row on the wall)."""
        for player in self.players:
            # Check rows
            for i in range(5):
                if all(player.board.occupancy[i, j] == 1 for j in range(5)):
                    return True
        return False

    def get_legal_moves(self) -> List[Tuple[str, int, Color, str, int]]:
        """Return list of legal moves as (source_type, source_index, color, destination_type, destination_index)."""
        moves = []
        player = self.players[self.current_player]
        
        # Helper to get available colors from a source
        def get_colors_from_source(source_type, source_index):
            if source_type == 'factory':
                factory = self.factories[source_index]
                colors = set(tile.color for tile in factory.tiles)
                return colors
            elif source_type == 'center':
                colors = set(tile.color for tile in self.center.tiles)
                return colors
            return set()
        
        # Check if a move is legal
        def is_legal_move(source_type, source_index, color, destination_type, destination_index):
            # Check if tiles are available in source
            available_colors = get_colors_from_source(source_type, source_index)
            if color not in available_colors:
                return False
            
            if destination_type == 'pattern_line':
                return player.pattern_lines.can_place(color, destination_index)
            elif destination_type == 'floor':
                return True  # Floor always accepts
            return False
        
        # Generate moves from factories
        for factory_idx in range(5):
            if self.factories[factory_idx].tiles:  # Only if factory has tiles
                available_colors = get_colors_from_source('factory', factory_idx)
                for color in available_colors:
                    # To pattern lines
                    for line_idx in range(5):
                        if is_legal_move('factory', factory_idx, color, 'pattern_line', line_idx):
                            moves.append(('factory', factory_idx, color, 'pattern_line', line_idx))
                    # To floor
                    moves.append(('factory', factory_idx, color, 'floor', None))
        
        # Generate moves from center
        if self.center.tiles:  # Only if center has tiles
            available_colors = get_colors_from_source('center', None)
            for color in available_colors:
                # To pattern lines
                for line_idx in range(5):
                    if is_legal_move('center', None, color, 'pattern_line', line_idx):
                        moves.append(('center', None, color, 'pattern_line', line_idx))
                # To floor
                moves.append(('center', None, color, 'floor', None))
        
        return moves

    def make_move(self, move: Tuple[str, int, Color, str, int]):
        """Apply a move to the game state."""
        source_type, source_index, color, destination_type, destination_index = move
        player = self.players[self.current_player]
        
        # Take tiles from source
        taken_tiles = []
        if source_type == 'factory':
            factory = self.factories[source_index]
            taken_tiles = [tile for tile in factory.tiles if tile.color == color]
            factory.tiles = [tile for tile in factory.tiles if tile.color != color]
            # Move remaining tiles to center
            for tile in factory.tiles:
                self.center.tiles.append(tile)
            factory.tiles = []
        elif source_type == 'center':
            taken_tiles = [tile for tile in self.center.tiles if tile.color == color]
            self.center.tiles = [tile for tile in self.center.tiles if tile.color != color]
            if self.center.has_first_player_marker:
                taken_tiles.append(FirstPlayerMarker())
                self.center.has_first_player_marker = False
                self.first_player_marker_holder = self.current_player
        
        # Place tiles in destination
        if destination_type == 'pattern_line':
            for tile in taken_tiles:
                if isinstance(tile, FirstPlayerMarker):
                    # Shift the floor to make space at index 0
                    # If floor[6] is occupied, discard it
                    if player.floor.tiles[6] is not None:
                        if not isinstance(player.floor.tiles[6], FirstPlayerMarker):
                            self.discard.add(player.floor.tiles[6])
                    # Shift tiles from 5 to 0 to 6 to 1
                    for i in range(6, 0, -1):
                        player.floor.tiles[i] = player.floor.tiles[i-1]
                    # Place marker at 0
                    player.floor.tiles[0] = tile
                elif player.pattern_lines.can_place(tile.color, destination_index):
                    player.pattern_lines.place_tile(tile, destination_index)
                else:
                    # Excess to floor
                    for i in range(7):
                        if player.floor.tiles[i] is None:
                            player.floor.tiles[i] = tile
                            break
                    else:
                        self.discard.add(tile)  # If floor is full, discard
        elif destination_type == 'floor':
            for tile in taken_tiles:
                if isinstance(tile, FirstPlayerMarker):
                    # Shift the floor to make space at index 0
                    # If floor[6] is occupied, discard it
                    if player.floor.tiles[6] is not None:
                        if not isinstance(player.floor.tiles[6], FirstPlayerMarker):
                            self.discard.add(player.floor.tiles[6])
                    # Shift tiles from 5 to 0 to 6 to 1
                    for i in range(6, 0, -1):
                        player.floor.tiles[i] = player.floor.tiles[i-1]
                    # Place marker at 0
                    player.floor.tiles[0] = tile
                else:
                    for i in range(7):
                        if player.floor.tiles[i] is None:
                            player.floor.tiles[i] = tile
                            break
                    else:
                        self.discard.add(tile)
        
        # Discard excess tiles if any (though in Azul, you take all of the color)
        # This is now handled above

    def get_winner(self):
        """Return the winner (player index) or None if tie."""
        if not self.is_terminal():
            return None
        scores = [player.score for player in self.players]
        max_score = max(scores)
        winners = [i for i, score in enumerate(scores) if score == max_score]
        if len(winners) == 1:
            return winners[0]
        return None  # Tie

    def refill_factories(self):
        """Refill factories with 4 tiles each from bag, shuffling discard back if needed."""
        for factory in self.factories:
            factory.tiles = []
            for _ in range(4):
                if self.bag.is_empty():
                    # Bag is empty, refill from discard
                    if not self.discard.is_empty():
                        self.bag.tiles.extend(self.discard.tiles)
                        self.discard.tiles = []
                        random.shuffle(self.bag.tiles)
                tile = self.bag.draw()
                if tile:
                    factory.tiles.append(tile)

    def end_round(self):
        """Handle end of round: move tiles from pattern lines to wall, calculate scores, handle floor penalties, and refill factories."""
        for player_idx, player in enumerate(self.players):
            # Process each pattern line
            for line_idx in range(5):
                line = player.pattern_lines.lines[line_idx]
                if player.pattern_lines.is_full(line_idx):
                    # Place the tile on the wall
                    tile = line[0]  # All tiles are the same color
                    color = tile.color
                    # Find the position in the wall for this color and row
                    for col in range(5):
                        if player.board.pattern[line_idx, col] == color:
                            if player.board.occupancy[line_idx, col] == 0:
                                player.board.occupancy[line_idx, col] = 1
                                # Score for placement
                                score = 1  # Base score
                                # Add adjacent tiles in row and column
                                row_score = sum(1 for c in range(5) if player.board.occupancy[line_idx, c] == 1) - 1
                                col_score = sum(1 for r in range(5) if player.board.occupancy[r, col] == 1) - 1
                                score += row_score + col_score
                                player.score += score
                                break
                    # Move remaining tiles to discard
                    remaining_tiles = line[1:]
                    for tile in remaining_tiles:
                        self.discard.add(tile)
                    # Remove tiles from pattern line (they are placed)
                    player.pattern_lines.lines[line_idx] = []
                else:
                    # Tiles stay in the pattern line until it is full
                    pass
            
            # Handle floor penalties
            for i in range(7):
                if player.floor.tiles[i] is not None:
                    player.score += player.floor.penalties[i]
            
            # Move floor tiles to discard
            for i in range(7):
                if player.floor.tiles[i] is not None:
                    if isinstance(player.floor.tiles[i], FirstPlayerMarker):
                        self.center.has_first_player_marker = True
                    else:
                        self.discard.add(player.floor.tiles[i])
            
            # Clear floor for next round
            player.floor.tiles = [None] * 7
        
        # Set first player for next round
        if self.first_player_marker_holder is not None:
            self.current_player = self.first_player_marker_holder
        
        # Refill factories for next round
        self.refill_factories()
        
        # Reset center (except first player marker if not taken)
        self.center.tiles = []
        if self.center.has_first_player_marker:
            # First player marker stays, but for simplicity, reset
            self.center.has_first_player_marker = True

    def copy(self):
        """Return a deep copy of the game state."""
        return copy.deepcopy(self)

class PatternLine:
    def __init__(self, board):
        self.lines = [[] for _ in range(5)]  # 5 lines for staging tiles
        self.board = board  # Reference to the player's board for restrictions

    def can_place(self, color: Color, line_index: int) -> bool:
        """Check if a tile of the given color can be placed in the specified line."""
        if not (0 <= line_index < 5):
            return False
        line = self.lines[line_index]
        # Check capacity (line_index + 1 spaces)
        if len(line) >= line_index + 1:
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

class FirstPlayerMarker:
    pass
