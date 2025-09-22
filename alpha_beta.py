import math
from azul_game import GameState

def evaluate_state(state):
    """Evaluate the game state: score difference for the current player."""
    if state.is_terminal():
        winner, bonuses = state.get_winner()
        scores = [player.score for player in state.players]
        if winner == state.current_player:
            return scores[state.current_player] - scores[1 - state.current_player]
        else:
            return -100  # Loss
    else:
        scores = [player.score for player in state.players]
        return scores[state.current_player] - scores[1 - state.current_player]

def move_heuristic(state, move):
    """Heuristic value for a move: score difference after the move."""
    temp = state.copy()
    original_score = temp.players[state.current_player].score
    temp.make_move(move)
    new_score = temp.players[state.current_player].score
    return new_score - original_score

def alpha_beta(state, depth, alpha, beta, maximizing_player, indent=0):
    """Alpha-Beta Pruning for Azul."""
    print("  " * indent + f"Alpha-Beta: depth={depth}, player={maximizing_player}, alpha={alpha}, beta={beta}")
    if depth == 0 or state.is_terminal():
        eval_val = evaluate_state(state)
        print("  " * indent + f"Leaf evaluation: {eval_val}")
        return eval_val
    
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        # No moves, end round
        temp_state = state.copy()
        temp_state.end_round()
        print("  " * indent + "No moves, ending round")
        return alpha_beta(temp_state, depth - 1, alpha, beta, maximizing_player, indent)
    
    # Sort moves by heuristic: high to low for maximizing, low to high for minimizing
    legal_moves = sorted(legal_moves, key=lambda m: move_heuristic(state, m), reverse=maximizing_player)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in legal_moves:
            temp_state = state.copy()
            temp_state.make_move(move)
            temp_state.current_player = 1 - temp_state.current_player
            if all(not f.tiles for f in temp_state.factories) and not temp_state.center.tiles:
                temp_state.end_round()
            eval = alpha_beta(temp_state, depth - 1, alpha, beta, False, indent + 1)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                print("  " * indent + "Beta cutoff!")
                break  # Beta cutoff
        print("  " * indent + f"Max returning: {max_eval}")
        return max_eval
    else:
        min_eval = math.inf
        for move in legal_moves:
            temp_state = state.copy()
            temp_state.make_move(move)
            temp_state.current_player = 1 - temp_state.current_player
            if all(not f.tiles for f in temp_state.factories) and not temp_state.center.tiles:
                temp_state.end_round()
            eval = alpha_beta(temp_state, depth - 1, alpha, beta, True, indent + 1)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                print("  " * indent + "Alpha cutoff!")
                break  # Alpha cutoff
        print("  " * indent + f"Min returning: {min_eval}")
        return min_eval

def get_best_move_alpha_beta(game_state, depth=2):
    """Get the best move using Alpha-Beta for the current player."""
    legal_moves = game_state.get_legal_moves()
    if not legal_moves:
        return None
    
    # Sort moves by heuristic (best first for maximizing)
    legal_moves = sorted(legal_moves, key=lambda m: move_heuristic(game_state, m), reverse=True)
    
    best_move = None
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf
    
    for move in legal_moves:
        print(f"Evaluating top-level move: {move}")
        temp_state = game_state.copy()
        temp_state.make_move(move)
        temp_state.current_player = 1 - temp_state.current_player
        if all(not f.tiles for f in temp_state.factories) and not temp_state.center.tiles:
            temp_state.end_round()
        move_value = alpha_beta(temp_state, depth - 1, alpha, beta, False, indent=0)
        print(f"Move {move} value: {move_value}")
        if move_value > best_value:
            best_value = move_value
            best_move = move
        alpha = max(alpha, move_value)
    
    print(f"Best move: {best_move} with value: {best_value}")
    return best_move
