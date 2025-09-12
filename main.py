from azul_game import GameState
from mcts import MCTS, Node

def main():
    # Initialize game
    game = GameState()
    game.refill_factories()  # Fill factories with tiles to start the game

    while not game.is_terminal():
        # Create MCTS for current player
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            # No legal moves, end the round
            game.end_round()
            continue  # Skip to next iteration
        
        root = Node(game)
        mcts = MCTS(root)
        mcts.search(iterations=1000)
        best_move = mcts.get_best_move()
        game.make_move(best_move)  # Apply best move
        game.current_player = 1 - game.current_player

    winner = game.get_winner()
    print(f"Winner: Player {winner}")

if __name__ == "__main__":
    main()
