from typing import List

from gsm.core.graph import Graph


def grid_to_graph(grid: List[List[int]]) -> Graph:
    """
    Convert a 2D int grid into a Graph:

    - nodes: (r, c) with attribute 'color'
    - edges: 4-neighbor adjacency
    """
    g = Graph()
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # nodes
    for r in range(rows):
        for c in range(cols):
            g.add_node((r, c), color=grid[r][c])

    # edges (4-neighborhood)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    g.add_edge((r, c), (rr, cc))

    return g


def graph_to_grid(g: Graph) -> List[List[int]]:
    """
    Convert a Graph with nodes (r, c) and attribute 'color' back to a 2D grid.

    Assumes coordinates start at 0 and form a dense rectangle.
    """
    if not g.nodes:
        return []

    max_r = max(r for (r, c) in g.nodes.keys())
    max_c = max(c for (r, c) in g.nodes.keys())
    rows = max_r + 1
    cols = max_c + 1

    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for (r, c), attrs in g.nodes.items():
        grid[r][c] = attrs["color"]

    return grid

