from alpha_beta_agent import AlphaBetaAgent

class MinimaxAgentWithPieceSquareTables(AlphaBetaAgent):
    def __init__(self, name, depth: int):
        super().__init__(name, depth)
        self.weights["piece_square"] = 1

    def name(self) -> str:
        return super().name() + "_with_piece_square_tables"

class KingSafetyAndMobility(AlphaBetaAgent):
    def __init__(self, name, depth: int):
        super().__init__(name, depth)
        self.weights["mobility_assess"] = 0.2
        self.weights["king_safety"] = 0.5
