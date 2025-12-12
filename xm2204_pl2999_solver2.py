#!/usr/bin/env python3
"""
CS 6613 - Project 2: Sudoku Mine
Backtracking + MRV + Degree + Forward Checking

Usage:
    python sudoku_mine.py input.txt output.txt

Input format (9 lines, each 9 ints 0..8, separated by spaces):
    0 = blank cell (no clue)
    1..8 = number of mines in its 8-neighbors (this cell itself cannot be a mine)

Output format:
    d          # depth (level) of goal node, root = level 0
    N          # total number of nodes generated
    9 lines of 0/1 (0 = no mine, 1 = mine)

Notes:
    - Mines can only be placed on cells where input is 0.
    - Each row, column, and 3x3 block must contain exactly 3 mines.
    - For a clue cell with value k, the 8-neighbor cells must have exactly k mines.
"""
import os
import sys
from typing import List, Optional, Tuple


GRID_SIZE = 9
BLOCK_SIZE = 3


class SudokuMineSolver:
    def __init__(self, clues: List[List[int]]):
        """
        clues[r][c] = 0   -> empty cell, can (potentially) contain a mine
        clues[r][c] = 1..8 -> clue; this cell itself is NOT a mine
        """
        self.clues = clues  # original numbers (0..8)

        # assignment[r][c]:
        #   None  -> unassigned variable (input 0)
        #   0     -> no mine
        #   1     -> mine
        self.assignment: List[List[Optional[int]]] = [
            [None] * GRID_SIZE for _ in range(GRID_SIZE)
        ]

        # cells that can (potentially) be mines: input == 0
        self.variables: List[Tuple[int, int]] = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.clues[r][c] == 0:
                    self.variables.append((r, c))
                else:
                    # clue cells can NEVER be mines
                    self.assignment[r][c] = 0

        # search stats
        self.node_count = 0
        self.goal_depth: Optional[int] = None

    # ---------- constraint helpers ----------

    def _check_row(self, r: int) -> bool:
        """Row r must have exactly 3 mines; forward checking."""
        mines = 0
        unassigned = 0
        for c in range(GRID_SIZE):
            val = self.assignment[r][c]
            if val == 1:
                mines += 1
            elif val is None:
                # only cells with clue == 0 can be mines
                if self.clues[r][c] == 0:
                    unassigned += 1

        # too many mines already
        if mines > 3:
            return False
        # even if all unassigned become mines, still < 3
        if mines + unassigned < 3:
            return False
        return True

    def _check_col(self, c: int) -> bool:
        """Column c must have exactly 3 mines; forward checking."""
        mines = 0
        unassigned = 0
        for r in range(GRID_SIZE):
            val = self.assignment[r][c]
            if val == 1:
                mines += 1
            elif val is None:
                if self.clues[r][c] == 0:
                    unassigned += 1

        if mines > 3:
            return False
        if mines + unassigned < 3:
            return False
        return True

    def _check_block(self, r: int, c: int) -> bool:
        """3x3 block containing (r, c) must have exactly 3 mines; forward checking."""
        r0 = (r // BLOCK_SIZE) * BLOCK_SIZE
        c0 = (c // BLOCK_SIZE) * BLOCK_SIZE
        mines = 0
        unassigned = 0
        for dr in range(BLOCK_SIZE):
            for dc in range(BLOCK_SIZE):
                rr = r0 + dr
                cc = c0 + dc
                val = self.assignment[rr][cc]
                if val == 1:
                    mines += 1
                elif val is None:
                    if self.clues[rr][cc] == 0:
                        unassigned += 1

        if mines > 3:
            return False
        if mines + unassigned < 3:
            return False
        return True

    def _check_all_rows_cols_blocks(self) -> bool:
        """Check all row/col/block constraints under forward checking."""
        for r in range(GRID_SIZE):
            if not self._check_row(r):
                return False
        for c in range(GRID_SIZE):
            if not self._check_col(c):
                return False
        # blocks: we only need to check top-left of each 3x3 block
        for br in range(0, GRID_SIZE, BLOCK_SIZE):
            for bc in range(0, GRID_SIZE, BLOCK_SIZE):
                # use any cell in this block, say (br, bc)
                if not self._check_block(br, bc):
                    return False
        return True

    def _neighbors_of(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return list of (rr, cc) that are 8-neighbors of (r, c)."""
        res: List[Tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    res.append((rr, cc))
        return res

    def _check_clue_constraints(self) -> bool:
        """
        For each clue cell with value k:
            sum over 8-neighbors' mines == k (forward checking: bounds).
        """
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                k = self.clues[r][c]
                if k == 0:
                    continue  # not a clue

                mines_assigned = 0
                can_still_be_mine = 0

                for rr, cc in self._neighbors_of(r, c):
                    val = self.assignment[rr][cc]
                    if val == 1:
                        mines_assigned += 1
                    elif val is None and self.clues[rr][cc] == 0:
                        # variable neighbor, may become 1 later
                        can_still_be_mine += 1
                    # clue cells around it are always 0 and fixed; ignore

                # too many mines already
                if mines_assigned > k:
                    return False
                # even if all remaining neighbors become mines, still too few
                if mines_assigned + can_still_be_mine < k:
                    return False

        return True

    def _is_consistent(self) -> bool:
        """Check all constraints with forward checking."""
        return self._check_all_rows_cols_blocks() and self._check_clue_constraints()

    # ---------- MRV + Degree heuristics ----------

    def _unassigned_variables(self) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for (r, c) in self.variables
            if self.assignment[r][c] is None
        ]

    def _degree_score(self, r: int, c: int) -> int:
        """
        Degree heuristic: number of *other* unassigned variables that share
        a row, column, or block with (r, c).
        """
        score = 0
        for (rr, cc) in self._unassigned_variables():
            if rr == r and cc == c:
                continue
            # same row / column / block
            same_row = (rr == r)
            same_col = (cc == c)
            same_block = (rr // BLOCK_SIZE == r // BLOCK_SIZE) and (
                cc // BLOCK_SIZE == c // BLOCK_SIZE
            )
            if same_row or same_col or same_block:
                score += 1
        return score

    def _select_unassigned_variable(self) -> Optional[Tuple[int, int]]:
        """
        SELECT-UNASSIGNED-VARIABLE using MRV + Degree heuristic.

        MRV: choose variable with smallest remaining domain size
             (domain is subset of {0,1} that keeps constraints consistent).
        Tie-breaker: variable with largest degree_score.
        """
        unassigned = self._unassigned_variables()
        if not unassigned:
            return None

        best_var: Optional[Tuple[int, int]] = None
        best_domain_size = 3  # max is 2
        best_degree = -1

        for (r, c) in unassigned:
            domain_size = 0
            # test value 0 and 1
            for v in (0, 1):
                self.assignment[r][c] = v
                if self._is_consistent():
                    domain_size += 1
                self.assignment[r][c] = None  # revert

            # If domain_size == 0 for any variable, current partial assignment is dead.
            # But we just use this info in MRV: this variable is the "most constrained".
            if domain_size == 0:
                # we can short-circuit and return this one,
                # backtracking will fail immediately when trying values.
                return (r, c)

            deg = self._degree_score(r, c)
            if (domain_size < best_domain_size) or (
                domain_size == best_domain_size and deg > best_degree
            ):
                best_domain_size = domain_size
                best_degree = deg
                best_var = (r, c)

        return best_var

    # ---------- backtracking search ----------

    def _backtrack(self, depth: int) -> bool:
        """
        Recursive backtracking.
        depth = level in search tree (root = 0).
        """
        self.node_count += 1

        # if all variables assigned -> success
        if not self._unassigned_variables():
            if self._is_consistent():
                self.goal_depth = depth
                return True
            return False

        var = self._select_unassigned_variable()
        if var is None:
            return False

        r, c = var

        # ORDER-DOMAIN-VALUES(S, A): here simply {0,1} (no LCV)
        for value in (0, 1):
            self.assignment[r][c] = value

            if self._is_consistent():
                # INFERENCE: we already did forward checking in _is_consistent()
                if self._backtrack(depth + 1):
                    return True

            # undo assignment
            self.assignment[r][c] = None

        return False

    def solve(self) -> bool:
        """
        Solve the Sudoku Mine puzzle.
        Returns True if solved, False otherwise.
        """
        solved = self._backtrack(depth=0)
        return solved

    # ---------- I/O helpers ----------

    def write_output(self, filename: str):
        """
        Write:
            d
            N
            9x9 assignment grid (0/1)
        """
        if self.goal_depth is None:
            d = -1
        else:
            d = self.goal_depth

        with open(filename, "w") as f:
            f.write(str(d) + "\n")
            f.write(str(self.node_count) + "\n")
            for r in range(GRID_SIZE):
                row_vals = []
                for c in range(GRID_SIZE):
                    val = self.assignment[r][c]
                    if val is None:
                        # should not happen if solved; treat as 0
                        val = 0
                    row_vals.append(str(val))
                f.write(" ".join(row_vals) + "\n")


def read_input_file(filename: str) -> List[List[int]]:
    """
    Read a 9x9 grid of ints (0..8), separated by spaces.
    """
    grid: List[List[int]] = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != GRID_SIZE:
                raise ValueError("Each line must have 9 integers.")
            row = [int(x) for x in parts]
            grid.append(row)

    if len(grid) != GRID_SIZE:
        raise ValueError("Input file must contain exactly 9 lines of numbers.")

    return grid

def main():
    input_folder = "inputs"
    output_folder = "outputs"

    # 確保 outputs 資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 讀取 inputs/ 裡所有 .txt 檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".txt", "_output.txt")
            output_path = os.path.join(output_folder, output_filename)

            print(f"Processing {input_path} -> {output_path}")

            clues = read_input_file(input_path)
            solver = SudokuMineSolver(clues)
            solved = solver.solve()

            if solved:
                print(f"✓ Solved {filename}")
            else:
                print(f"✗ No solution for {filename}")

            solver.write_output(output_path)


if __name__ == "__main__":
    main()