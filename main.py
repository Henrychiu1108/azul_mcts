from azul_game import GameState
from mcts import MCTS, Node
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

def main():
    # Redirect output to file
    original_stdout = sys.stdout
    with open('game_output.txt', 'w') as f:
        sys.stdout = f
        # Initialize game
        game = GameState()
        game.refill_factories()  # Fill factories with tiles to start the game
        print("Game initialized.")
        print_game_state(game)

        round_count = 0
        while not game.is_terminal() and round_count < 1:
            print(f"\n--- Player {game.current_player}'s Turn ---")
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                print("No legal moves available. Ending round.")
                game.end_round()
                print("Round ended.")
                print_game_state(game)
                round_count += 1
                continue
            
            print("Legal Moves:")
            for idx, move in enumerate(legal_moves):
                source_type, source_index, color, dest_type, dest_index = move
                print(f"  {idx}: {source_type} {source_index} -> {dest_type} {dest_index} ({color.value})")
            
            # For testing, choose a random move from legal moves
            chosen_idx = random.randint(0, len(legal_moves) - 1)
            if chosen_idx < len(legal_moves):
                move = legal_moves[chosen_idx]
                print(f"Chosen Move: {move}")
                game.make_move(move)
                print("Move applied.")
                print_game_state(game)
            else:
                print("Invalid choice.")
            
            # Switch player
            game.current_player = 1 - game.current_player

        winner = game.get_winner()
        print(f"\nGame Over. Winner: Player {winner}")
    # Restore stdout
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
