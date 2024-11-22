import chess

def evaluate_mobility(board: chess.Board):
    """
    mobility WRT legal moves; returns positive int if active has more legal moves
    """
    own_mobility = len(list(board.legal_moves)) #active player legal moves then switch to opponent
    board.push(chess.Move.null()) #null move switch turns without alteration
    opponent_mobility = len(list(board.legal_moves)) #opponent's legal moves
    board.pop() #remove null move switch to restore
    return own_mobility - opponent_mobility #positive=more mobility than opponent

