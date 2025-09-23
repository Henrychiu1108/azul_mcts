from azul_game import GameState
from mcts import MCTS, Node
from alpha_beta import get_best_move_alpha_beta
import random
import sys

def print_game_state(game):
    """Print the current game state for debugging."""
    print(f"\nCurrent Player: {game.current_player}")
    print(f"First Player Marker Holder: {game.first_player_marker_holder}")
    print("Factories:")
    for i, factory in enumerate(game.factories):
        print(f"  Factory {i}: {[tile.color.value for tile in factory.tiles]}")
    print(f"Center: {[tile.color.value for tile in game.center.tiles]} (Marker: {game.center.has_first_player_marker})")
    print("Players:")
    for i, player in enumerate(game.players):
        print(f"  Player {i} Score: {player.score}")
        print(f"    Pattern Lines:")
        for j, line in enumerate(player.pattern_lines.lines):
            print(f"      Line {j}: {[tile.color.value for tile in line]}")
        print(f"    Floor: {[tile.__class__.__name__ if tile is not None else None for tile in player.floor.tiles]}")
        print(f"    Board Occupancy:\n{player.board.occupancy}")
    print(f"Bag: {len(game.bag.tiles)} tiles")
    print(f"Discard: {len(game.discard.tiles)} tiles")

def mcts(game):
    root = Node(game.copy())
    mcts = MCTS(root)
    mcts.search(iterations=100)  # Adjust iterations as needed
    best_move = mcts.get_best_move()
    return best_move

def alpha_beta(game_state, depth=10):
    best_move = get_best_move_alpha_beta(game_state, depth)
    return best_move

def is_last_round(game) -> bool:
    # Check if this is the last round (any player has a row with 4 tiles)
    return any(
        sum(player.board.occupancy[row]) == 4
        for player in game.players
        for row in range(5)
    )

def main():
    # Redirect output to file
    original_stdout = sys.stdout
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        # Initialize game
        game = GameState()
        game.refill_factories()  # Fill factories with tiles to start the game
        print("Game initialized.")
        round_number = 1
        print_game_state(game)

        while not game.is_terminal():
            print(f"\n--- Player {game.current_player}'s Turn ---")
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.end_round()
                print("Round ended.")
                print_game_state(game)
                continue
            if is_last_round(game):
                print("Last round detected. Running Alpha-Beta search for both players.")
                best_move = get_best_move_alpha_beta(game, depth=10)
                game.make_move(best_move)
                print(f"Current player move: {best_move}")
            else:
                best_move = mcts(game)
            game.make_move(best_move)
            print_game_state(game)
            # 回合結束由下一輪 while 開頭偵測空 moves

        winner = game.get_winner()
        print(f"\nGame Over. Winner: Player {winner}")
    # Restore stdout
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
