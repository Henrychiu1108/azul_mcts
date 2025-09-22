import math
import random
from azul_game import GameState

class Node:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = move

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c=1.4):
        choices_weights = [
            (child.value / child.visits if child.visits > 0 else 0) + c * math.sqrt(2 * math.log(max(1, self.visits)) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            if not any(child.move == move for child in self.children):
                new_state = self.state.copy()
                new_state.make_move(move)
                child_node = Node(new_state, self, move)
                self.children.append(child_node)
                return child_node
        return None

class MCTS:
    def __init__(self, root: Node):
        self.root = root
        self.root_player = root.state.current_player

    def search(self, iterations=50000):
        # Normal MCTS (removed last-turn special-case heuristic)
        for _ in range(iterations):
            node = self.select(self.root)
            if node.state.is_terminal():
                reward = self.simulate(node)
            else:
                expanded = node.expand()
                if expanded:
                    reward = self.simulate(expanded)
                else:
                    reward = self.simulate(node)
            self.backpropagate(node, reward)

    def select(self, node: Node):
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
        return node

    def simulate(self, node: Node):
        state = node.state
        
        # Heuristic scoring for moves
        def heuristic_score(move):
            dest_type = move[3]
            if dest_type == 'floor':
                return 0.1  # Low priority for floor
            elif dest_type == 'pattern_line':
                dest_index = move[4]
                color = move[2]
                player = state.players[state.current_player]
                # Find the column for this color in the row
                column = None
                for col in range(5):
                    if player.board.pattern[dest_index, col] == color:
                        column = col
                        break
                if column is not None:
                    wall = player.board.occupancy
                    if wall[dest_index][column] == 0:  # If not already occupied
                        # Simulate placement
                        temp_wall = wall.copy()
                        temp_wall[dest_index][column] = 1
                        # Check if row is full
                        if all(temp_wall[dest_index]):
                            return 20  # High bonus for completing a wall row
                        # Check if column is full
                        if all(temp_wall[i][column] for i in range(5)):
                            return 20  # High bonus for completing a wall column
                        # Check if this move provides the missing color for a row with 4 tiles
                        for row in range(5):
                            if sum(wall[row]) == 4 and wall[row][column] == 0:
                                return 15  # High bonus for completing a row with 4 tiles
                # Prefer lower rows (easier to complete) if not completing
                return 1 + (4 - dest_index)
            return 1
        
        # Random playout with heuristic bias
        while not state.is_terminal():
            moves = state.get_legal_moves()
            if not moves:
                state.end_round()
                continue
            
            scores = [heuristic_score(m) for m in moves]
            move = random.choices(moves, weights=scores, k=1)[0]
            
            state.make_move(move)
            # Switch player
            state.current_player = 1 - state.current_player
            # Check if round should end after move
            if all(not factory.tiles for factory in state.factories) and not state.center.tiles:
                state.end_round()
        winner = state.get_winner()
        scores = [player.score for player in state.players]
        return scores[self.root_player] - scores[1 - self.root_player]

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
