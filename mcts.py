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
        # 未嘗試之合法手列表（增量擴展）
        self.untried_moves = state.get_legal_moves()

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
        new_state = self.state.copy()
        new_state.make_move(move)
        # 回合結束改以後續空合法手判斷
        if not new_state.get_legal_moves():
            new_state.end_round()
        elif not new_state.is_terminal():
            new_state.switch_player()
        child_node = Node(new_state, self, move)
        self.children.append(child_node)
        return child_node

class MCTS:
    def __init__(self, root: Node):
        self.root = root
        self.root_player = root.state.current_player

    def search(self, iterations=50000):
        for _ in range(iterations):
            node = self.select(self.root)
            expanded = node.expand()
            node_to_sim = expanded if expanded else node
            reward = self.simulate(node_to_sim)
            self.backpropagate(node_to_sim, reward)

    def select(self, node: Node):
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
        return node

    def simulate(self, node: Node):
        state = node.state

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

        while not state.is_terminal():
            moves = state.get_legal_moves()
            if not moves:
                state.end_round()
                continue
            scores = [heuristic_score(m) for m in moves]
            move = random.choices(moves, weights=scores, k=1)[0]
            state.make_move(move)
            if not state.get_legal_moves():
                state.end_round()
            elif not state.is_terminal():
                state.switch_player()

        winner = state.get_winner()
        scores = [p.score for p in state.players]
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
