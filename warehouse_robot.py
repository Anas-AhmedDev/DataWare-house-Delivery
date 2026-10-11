# Warehouse Delivery Robot - Intelligent Systems, Task 1: Search Algorithms
# Student: Anas Ahmed Saad  |  Student ID: 2403244286
# Algorithms: Uniform Cost Search (UCS) and A* Search
# Only the standard library is used (heapq and time).

import heapq
import time

# ---------------------------------------------------------------
# MAPS (each map is a list of strings, one string per row)
# ---------------------------------------------------------------
MAP_A = [
    "S..#....",
    "##.#.##.",
    "....~~..",
    ".####.#.",
    "...~..#G"
]

MAP_B = [
    "S~~~~~~~G",
    ".#######.",
    ".........",
    ".#.#.#.#.",
    "........."
]

MAP_C = [
    "S..#....",
    ".#.#.##.",
    "...#..#.",
    ".###.##.",
    "....#..G"
]

GRID = MAP_A   # store the map in a variable called GRID (used for a single-map test)

MOVES = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1)    # Right
]


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def find_start_goal(grid):
    """Scan the grid to find S and G (their positions are not typed by hand)."""
    start = None
    goal = None
    for row in range(len(grid)):                # visit every ROW (outer loop)
        for col in range(len(grid[row])):       # visit every COLUMN (inner loop)
            cell = grid[row][col]               # the character at this row and column
            if cell == "S":
                start = (row, col)
            if cell == "G":
                goal = (row, col)
    return start, goal


def get_neighbors(grid, position):
    """Return the cells the robot can move to (Up, Down, Left, Right, no walls)."""
    neighbors = []
    row = position[0]
    col = position[1]
    for move in MOVES:
        new_row = row + move[0]   # row after the move
        new_col = col + move[1]   # column after the move
        if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[new_row]):  # inside the map?
            if grid[new_row][new_col] != "#":                               # not a wall?
                neighbors.append((new_row, new_col))
    return neighbors


def get_cost(grid, position):
    """Cost of ENTERING a cell: rough floor '~' = 3, everything else (. and G) = 1."""
    row = position[0]
    col = position[1]
    if grid[row][col] == "~":   # rough floor
        return 3
    return 1                    # normal floor or goal


def manhattan(position, goal):
    """Heuristic for A*: abs(row difference) + abs(column difference)."""
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def build_path(came_from, goal):
    """Follow came_from from the goal back to the start, then reverse the list."""
    path = []
    position = goal
    while position is not None:
        path.append(position)
        position = came_from[position]
    path.reverse()          # now the path goes from start to goal
    return path


def show_path(grid, path):
    """Print the map with * on the path. S and G stay unchanged."""
    for row in range(len(grid)):
        line = ""
        for col in range(len(grid[row])):
            symbol = grid[row][col]
            if (row, col) in path and symbol != "S" and symbol != "G":
                symbol = "*"
            line = line + symbol
        print("   " + line)


def make_result(name, path, cost, expanded, max_frontier, start_time):
    """Collect all the measurements in one dictionary."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return {"algorithm": name, "path": path, "cost": cost, "expanded": expanded,
            "max_frontier": max_frontier, "time_ms": elapsed_ms}


# ---------------------------------------------------------------
# UNIFORM COST SEARCH (priority = cost so far)
# ---------------------------------------------------------------
def ucs(grid, start, goal):
    start_time = time.perf_counter()
    frontier = [(0, start)]     # priority queue with (cost, position)
    came_from = {start: None}   # dictionary to track the path
    cost_so_far = {start: 0}    # cheapest known cost to reach each position
    expanded = 0                # counter for expanded nodes
    max_frontier = 1            # biggest size the frontier reached

    while frontier:
        current_cost, current = heapq.heappop(frontier)   # position with the lowest cost
        if current_cost != cost_so_far[current]:          # outdated entry -> skip it
            continue
        expanded += 1

        if current == goal:                               # goal reached
            path = build_path(came_from, goal)
            return make_result("UCS", path, cost_so_far[goal], expanded, max_frontier, start_time)

        for neighbor in get_neighbors(grid, current):
            new_cost = current_cost + get_cost(grid, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))   # priority = g

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    # the frontier is empty and the goal was never reached
    return make_result("UCS", None, None, expanded, max_frontier, start_time)


# ---------------------------------------------------------------
# A* SEARCH (priority = f = g + h)
# ---------------------------------------------------------------
def a_star(grid, start, goal):
    start_time = time.perf_counter()
    frontier = [(manhattan(start, goal), 0, start)]   # (f, g, position)
    came_from = {start: None}
    cost_so_far = {start: 0}
    expanded = 0
    max_frontier = 1

    while frontier:
        f, current_cost, current = heapq.heappop(frontier)
        if current_cost != cost_so_far[current]:      # outdated entry -> skip it
            continue
        expanded += 1

        if current == goal:
            path = build_path(came_from, goal)
            return make_result("A*", path, cost_so_far[goal], expanded, max_frontier, start_time)

        for neighbor in get_neighbors(grid, current):
            new_cost = current_cost + get_cost(grid, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + manhattan(neighbor, goal)   # f = g + h
                heapq.heappush(frontier, (priority, new_cost, neighbor))

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return make_result("A*", None, None, expanded, max_frontier, start_time)


# ---------------------------------------------------------------
# PRINTING AND RUNNING
# ---------------------------------------------------------------
def print_result(grid, result):
    print("Algorithm:", result["algorithm"])
    if result["path"] is None:
        print("No path found")
    else:
        print("Path:", result["path"])
        print("Total Cost:", result["cost"])
        print("Path length (moves):", len(result["path"]) - 1)
    print("Nodes expanded:", result["expanded"])
    print("Maximum frontier size:", result["max_frontier"])
    print("Run time: {:.4f} ms".format(result["time_ms"]))
    if result["path"] is not None:
        print("Map with path:")
        show_path(grid, result["path"])
    print()


def run_map(map_name, grid, all_results):
    """Run UCS and A* on one map and save the results for the table."""
    start, goal = find_start_goal(grid)
    print("=" * 50)
    print(map_name, "  Start:", start, " Goal:", goal)
    print("=" * 50)
    for result in [ucs(grid, start, goal), a_star(grid, start, goal)]:
        print_result(grid, result)
        result["map"] = map_name
        all_results.append(result)


def print_table(all_results):
    header = "{:<7}{:<11}{:<8}{:<8}{:<7}{:<10}{:<10}{:<10}".format(
        "Map", "Algorithm", "Found?", "Length", "Cost", "Expanded", "MaxFront", "Time(ms)")
    print("RESULTS TABLE")
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r["path"] is None:
            found, length, cost = "No", "-", "-"
        else:
            found, length, cost = "Yes", len(r["path"]) - 1, r["cost"]
        print("{:<7}{:<11}{:<8}{:<8}{:<7}{:<10}{:<10}{:<10}".format(
            r["map"], r["algorithm"], found, length, cost,
            r["expanded"], r["max_frontier"], "{:.4f}".format(r["time_ms"])))
    print()


def print_comparison(all_results):
    print("COMPARISON")
    for i in range(0, len(all_results), 2):      # results come in pairs: UCS, A*
        u = all_results[i]
        a = all_results[i + 1]
        print(u["map"] + ":")
        if u["path"] is None and a["path"] is None:
            print("   No solution exists, so neither algorithm found a path.")
        elif u["cost"] == a["cost"]:
            print("   Both found the same cheapest cost:", u["cost"])
        else:
            print("   Different costs! UCS =", u["cost"], " A* =", a["cost"])
        if a["expanded"] < u["expanded"]:
            print("   A* expanded fewer nodes:", a["expanded"], "vs", u["expanded"])
        elif a["expanded"] > u["expanded"]:
            print("   UCS expanded fewer nodes:", u["expanded"], "vs", a["expanded"])
        else:
            print("   Both expanded the same number of nodes:", u["expanded"])
    print()


# ---------------------------------------------------------------
# MAIN PART
# ---------------------------------------------------------------
all_results = []

# To test ONE map only: change GRID near the top (GRID = MAP_B), then use this line
# and put a # in front of the three run_map lines below it.
# run_map("GRID", GRID, all_results)

# Run all three maps (needed for the assignment)
run_map("MAP_A", MAP_A, all_results)
run_map("MAP_B", MAP_B, all_results)
run_map("MAP_C", MAP_C, all_results)

print_table(all_results)
print_comparison(all_results)
