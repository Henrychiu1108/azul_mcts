import math
import random
from azul_game import GameState

class Node:
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c=1.4):
        choices_weights = [
            (child.value / child.visits) + c * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            if not any(child.state == self.state.make_move(move) for child in self.children):
                new_state = self.state.copy()
                new_state.make_move(move)
                child_node = Node(new_state, self)
                self.children.append(child_node)
                return child_node
        return None

class MCTS:
    def __init__(self, root: Node):
        self.root = root

    def search(self, iterations=1000):
        for _ in range(iterations):
            node = self.select(self.root)
            if not node.state.is_terminal():
                node = node.expand()
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)

    def select(self, node: Node):
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
        return node

    def simulate(self, state: GameState):
        # Random playout
        while not state.is_terminal():
            moves = state.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
            state.make_move(move)
        return state.get_winner()  # Assume returns 1 for win, 0 for loss

    def backpropagate(self, node: Node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_best_move(self):
        return self.root.best_child(c=0).state
