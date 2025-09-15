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
            (child.value / child.visits if child.visits > 0 else 0) + c * math.sqrt(2 * math.log(self.visits) / (child.visits + 1e-6))
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

    def search(self, iterations=1000):
        for _ in range(iterations):
            node = self.select(self.root)
            if node.state.is_terminal():
                reward = self.simulate(node.state)
            else:
                expanded = node.expand()
                if expanded:
                    reward = self.simulate(expanded.state)
                else:
                    reward = self.simulate(node.state)
            self.backpropagate(node, reward)

    def select(self, node: Node):
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
        return node

    def simulate(self, state: GameState):
        # Random playout with bias towards pattern lines
        while not state.is_terminal():
            moves = state.get_legal_moves()
            if not moves:
                state.end_round()
                continue
            # Prefer pattern_line moves over floor moves
            pattern_moves = [m for m in moves if m[3] == 'pattern_line']
            if pattern_moves:
                move = random.choice(pattern_moves)
            else:
                move = random.choice(moves)
            state.make_move(move)
            # Switch player
            state.current_player = 1 - state.current_player
            # Check if round should end after move
            if all(not factory.tiles for factory in state.factories) and not state.center.tiles:
                state.end_round()
        winner, bonuses = state.get_winner()
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
