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

        while not game.is_terminal():
            print(f"\n--- Player {game.current_player}'s Turn ---")
            legal_moves = game.get_legal_moves()
            
            # print("Legal Moves:")
            # for idx, move in enumerate(legal_moves):
            #     source_type, source_index, color, dest_type, dest_index = move
            #     print(f"  {idx}: {source_type} {source_index} -> {dest_type} {dest_index} ({color.value})")
            
            # For testing, choose a random move from legal moves
            # Use MCTS to choose the best move
            root = Node(game.copy())
            mcts = MCTS(root)
            mcts.search(iterations=100)  # Adjust iterations as needed
            best_move = mcts.get_best_move()
            if best_move:
                move = best_move
                print(f"MCTS Chosen Move: {move}")
            else:
                if legal_moves:
                    chosen_idx = random.randint(0, len(legal_moves) - 1)
                    move = legal_moves[chosen_idx]
                    print(f"Random Chosen Move: {move}")
                else:
                    print("No legal moves, skipping turn.")
                    continue
            game.make_move(move)
            print("Move applied.")
            print_game_state(game)
            
            # Check if round should end: all factories empty and center empty
            if all(not factory.tiles for factory in game.factories) and not game.center.tiles:
                print("All sources empty. Ending round.")
                game.end_round()
                print("Round ended.")
                print_game_state(game)
            else:
                # Switch player
                game.current_player = 1 - game.current_player

        result = game.get_winner()
        if isinstance(result, tuple):
            winner, bonuses = result
            for idx, (row_b, col_b, color_b) in enumerate(bonuses):
                print(f"Player {idx} bonuses: {row_b} rows (+{row_b*2}), {col_b} columns (+{col_b*7}), {color_b} colors (+{color_b*10})")
        else:
            winner = result
        print(f"\nGame Over. Winner: Player {winner}")
    # Restore stdout
    sys.stdout = original_stdout
    print("Output written to game_output.txt")

if __name__ == "__main__":
    main()
