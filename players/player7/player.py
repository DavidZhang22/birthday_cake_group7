from shapely import Point, LineString
from players.player import Player, PlayerException
from src.cake import Cake
import itertools


class Player7(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def find_best_greedy_cut(self) -> tuple[Point, Point]:
        """Find the best cut using a greedy strategy.
        
        Algorithm:
        1. Find the piece with the highest interior ratio (most filling)
        2. Try all possible cuts on that piece
        3. Select the cut that maximizes the minimum interior ratio of the resulting pieces
        """
        pieces = self.cake.get_pieces()
        
        # Find the piece with the highest interior ratio
        best_piece = None
        best_ratio = -1
        
        for piece in pieces:
            ratio = self.cake.get_piece_ratio(piece)
            if ratio > best_ratio:
                best_ratio = ratio
                best_piece = piece
        
        if best_piece is None:
            # Fallback to largest piece if no piece has interior
            best_piece = max(pieces, key=lambda p: p.area)
        
        # Generate all possible cuts on this piece
        vertices = list(best_piece.exterior.coords[:-1])
        lines = [
            LineString([vertices[i], vertices[(i + 1) % len(vertices)]]) 
            for i in range(len(vertices))
        ]
        
        best_cut = None
        best_score = -1
        
        # Try all combinations of line centroids as cut endpoints
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                from_p = lines[i].centroid
                to_p = lines[j].centroid
                
                # Check if this cut is valid
                is_valid, reason = self.cake.cut_is_valid(from_p, to_p)
                if not is_valid:
                    continue
                
                # Simulate the cut to evaluate its quality
                cut_pieces = self.cake.cut_piece(best_piece, from_p, to_p)
                
                # Calculate the score: minimum interior ratio of the two pieces
                # This ensures both pieces get a fair amount of filling
                ratios = [self.cake.get_piece_ratio(piece) for piece in cut_pieces]
                score = min(ratios)  # Greedy: maximize the worst case
                
                if score > best_score:
                    best_score = score
                    best_cut = (from_p, to_p)
        
        if best_cut is None:
            # If no valid cut found, try a more exhaustive search
            return self.find_fallback_cut()
        
        return best_cut
    
    def find_fallback_cut(self) -> tuple[Point, Point]:
        """Fallback method when greedy strategy fails."""
        # Find the largest piece
        largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
        vertices = list(largest_piece.exterior.coords[:-1])
        lines = [
            LineString([vertices[i], vertices[(i + 1) % len(vertices)]]) 
            for i in range(len(vertices))
        ]
        
        # Try all possible cuts
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                from_p = lines[i].centroid
                to_p = lines[j].centroid
                
                is_valid, reason = self.cake.cut_is_valid(from_p, to_p)
                if is_valid:
                    return (from_p, to_p)
        
        # If line centroids don't work, try vertices directly
        for i in range(len(vertices)):
            for j in range(i + 2, len(vertices)):  # Skip adjacent vertices
                from_p = Point(vertices[i])
                to_p = Point(vertices[j])
                
                is_valid, reason = self.cake.cut_is_valid(from_p, to_p)
                if is_valid:
                    return (from_p, to_p)
        
        raise PlayerException("Could not find any valid cut")

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Generate all cuts needed for the children using greedy strategy."""
        moves: list[tuple[Point, Point]] = []
        
        for i in range(self.children - 1):
            from_p, to_p = self.find_best_greedy_cut()
            moves.append((from_p, to_p))
            
            # Apply the cut to our cake representation
            self.cake.cut(from_p, to_p)
        
        return moves
