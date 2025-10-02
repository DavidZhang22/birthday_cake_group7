from shapely import Point, LineString, Polygon
from shapely.ops import split

from players.player import Player
from src.cake import Cake

import math

class Player7(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.cuts = []
        self.target_size = cake.exterior_shape.area / children
        print(f"Target size: {self.target_size}")



    def sample_on_edges(self, a, b, step=1, include_start=True):

        (x1, y1), (x2, y2) = a, b
        dx, dy = x2 - x1, y2 - y1

        if dx == 0 and dy == 0:
            return []
        
        n = max(2, math.ceil(math.hypot(dx, dy)/step))

        start_t = 0.0 if include_start else (1.0 / (n + 1))
        end_t   = 1.0

        pts = [start_t + i * (end_t - start_t) / (n - 1) for i in range(int(n))]
        pts = [(x1 + t * dx, y1 + t * dy) for t in pts]

        return pts

    def area_balance_score(self, poly: Cake, p1: Point, p2: Point):
        pieces = poly.cut_piece(poly.exterior_shape, p1, p2)
        assert len(pieces) == 2
       
        a, b = pieces[0].area, pieces[1].area
        
        return min(abs(a - self.target_size), abs(b - self.target_size))

    def pick_best_cut(self, piece: Cake, step=0.2, score_fn=None):

        if score_fn is None:
            score_fn = self.area_balance_score

        outline = list(piece.exterior_shape.exterior.coords[:-1])
        m = len(outline)

        edge_points = []
        for i in range(m):
            a = outline[i]
            b = outline[(i + 1) % m]
            # Include start of edge, exclude end to avoid duplicates at vertices
            samples = self.sample_on_edges(a, b, step=step, include_start=True)
            for pt in samples:
                edge_points.append((i, pt))

        # Edge case: ensure we at least have the original vertices if the edges are too short
        if not edge_points:
            for i in range(m):
                edge_points.append((i, outline[i]))
        
        print(len(edge_points), "edge points sampled")

        best_pair = None
        best_score = float("inf")
        for ei, (xi, yi) in edge_points:
            p1 = Point(xi, yi)
            for ej, (xj, yj) in reversed(edge_points):
                if ej == ei:
                    break
                p2 = Point(xj, yj)

                is_valid, _ = piece.cut_is_valid(p1, p2)
                if not is_valid:
                    continue
                
                
                s = score_fn(piece, p1, p2)
                if s < best_score:
                    best_score = s
                    best_pair = (p1, p2)

        return best_pair



    def get_cuts(self) -> list[tuple[Point, Point]]:

        while(len(self.cuts) < self.children - 1):
            largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
            cut = self.pick_best_cut(Cake(largest_piece, self.children - len(self.cuts), True), step=0.4)
            if cut is None:
                break
            self.cake.cut(cut[0], cut[1])
            self.cuts.append(cut)

        return self.cuts
