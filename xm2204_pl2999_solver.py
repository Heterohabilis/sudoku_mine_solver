import os
import sys

DOMAIN = [0, 1]


class Variable:
    """
        Variable Object to represent the Variables in the CSP.
        Attributes:
            row: Row index of the variable in the grid.
            col: Column index of the variable in the grid.
            domain: Set of possible values (0 or 1) for the variable.
            neighbors: Set of neighboring Variable objects that share constraints.
        
        Methods:
            __init__(self, row, col): Initializes the Variable with its position and full domain
            __repr__(self): Returns a string representation of the Variable (for debugging only).
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.domain = set(DOMAIN)
        self.neighbors = set()

    def __repr__(self):
        return (f"Variable({self.row}, {self.col}, "+
        f"domain={self.domain}), neighbors={len(self.neighbors)}")


def parse_input(ifle):
    """
        Parses the input file and returns variables and number_dicts.
        Args:
            ifle (str): Path to the input file.
        Read the input file and consturct two dictionaries:
            variables: A dictionary mapping (row, col) to Variable objects for empty cells.
            number_dicts: A dictionary mapping (row, col) to integers for numbered cells, where the integer represents the count of adjacent mines.
        Returns: the two dictionaries.
    """
    with open(ifle, 'r') as f:
        _lines = f.readlines()

    lines = [line.strip().split() for line in _lines]
        
    variables, number_dicts = {}, {}
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] == '0':  # 0 means a variable cell
                variables[(i, j)] = Variable(i, j)
            else:
                number_dicts[(i, j)] = int(lines[i][j])
    return variables, number_dicts


def build_constraint_graph(variables_dict, number_dicts):
    """
        Builds the constraint graph by establishing neighbor relationships between variables.
        Args:
            variables_dict (dict): Dictionary mapping (row, col) to Variable objects.
            number_dicts (dict): Dictionary mapping (row, col) to integers for numbered cells
        Check in 4 ways:
            - Sudoku Checking:
                - Same Row
                - Same Column
                - Same 3x3 Block
            - Number Cell Checking: Check if two variables are controlled by the same number cell.
        Returns: None (modifies the variable objects in place).
    """
    number_cell_map = {}
    for (nr, nc), number in number_dicts.items():
        number_cell_map[(nr, nc)] = set()
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            neighbor_pos = (nr + dr, nc + dc)
            if neighbor_pos in variables_dict:
                number_cell_map[(nr, nc)].add(variables_dict[neighbor_pos])

    all_vars_list = list(variables_dict.values())
    
    for i in range(len(all_vars_list)):
        var1 = all_vars_list[i]
        for j in range(i + 1, len(all_vars_list)):
            var2 = all_vars_list[j]
            
            is_neighbor = False
            
            if var1.row == var2.row: is_neighbor = True
            elif var1.col == var2.col: is_neighbor = True
            elif (var1.row // 3 == var2.row // 3) and (var1.col // 3 == var2.col // 3):
                is_neighbor = True
            
            if not is_neighbor:
                for affected_vars_set in number_cell_map.values():
                    if var1 in affected_vars_set and var2 in affected_vars_set:
                        is_neighbor = True
                        break 
            
            if is_neighbor:
                var1.neighbors.add(var2)
                var2.neighbors.add(var1)


def _mrv(variables):
    """
        Minimum Remaining Values (MRV) heuristic to select variables with the smallest domain size.
        Args:
            variables (list): List of Variable objects.
        Returns: List of Variable objects with the smallest domain size.
        Note that variables list is all unassigned variables, which is filtered by the caller of the function.
    """
    _min = min(variables, key=lambda var: len(var.domain))
    res = []
    for var in variables:
        if len(var.domain) == len(_min.domain):
            res.append(var)
    return res


def _degree(curr_assignment, vars_after_min):
    """
        Degree heuristic to select the variable with the most unassigned neighbors.
        Args:
            curr_assignment (dict): Current assignment of variables.
            vars_after_min (list): List of Variable objects after applying MRV.
        Returns: The Variable object with the highest degree (most unassigned neighbors).
    """
    if len(vars_after_min) == 1: return vars_after_min[0]
    res, curr_max_unasign = None, -1
    for var in vars_after_min:
        curr = 0
        for nbr in var.neighbors:
            if (nbr.row, nbr.col) not in curr_assignment:
                curr+=1
        if curr > curr_max_unasign:
            curr_max_unasign = curr
            res = var
    return res


def select_unassigned_var(curr_assignment, vars):
    """
        Selects the next variable to assign using MRV and Degree heuristics. Filter out already assigned variables, and then pass the list to MRV and Degree functions.
        Args:
            curr_assignment (dict): Current assignment of variables.
            vars (dict): Dictionary mapping (row, col) to Variable objects.
        Returns: The selected Variable object to assign next.

    """
    var_lst = []
    for k, v in vars.items():
        if k not in curr_assignment:
            var_lst.append(v)
    after_mrv = _mrv(var_lst)
    next = _degree(curr_assignment, after_mrv)
    return next


def is_consistent(assignment, number_dicts, all_variables_set):
    """
        Checks if the current assignment is consistent with Sudoku and number cell constraints.
        Check in four ways: sudoku (three ways) and number cell checking.

        Args:
            assignment (dict): Current assignment of variables.
            number_dicts (dict): Dictionary mapping (row, col) to integers for numbered cells.
            all_variables_set (set): Set of all variable positions (row, col).
        Returns: True if the assignment is consistent, False otherwise.
    """
    for r in range(9):
        mines_in_row = 0
        unassigned_in_row = 0
        
        for c in range(9):
            pos = (r, c)
            if pos in all_variables_set: 
                if pos in assignment:
                    mines_in_row += assignment[pos] 
                else:
                    unassigned_in_row += 1 
        
        if mines_in_row > 3:    # >3 mines in row, cannot place mine anymore
            return False
        if mines_in_row + unassigned_in_row < 3:    # <3, must place mine in ALL unassigned cells
            return False

    for c in range(9):
        mines_in_col = 0
        unassigned_in_col = 0
        
        for r in range(9):
            pos = (r, c)
            if pos in all_variables_set:
                if pos in assignment:
                    mines_in_col += assignment[pos]
                else:
                    unassigned_in_col += 1
        
        if mines_in_col > 3:
            return False
        if mines_in_col + unassigned_in_col < 3:
            return False

    for block_idx in range(9):
        mines_in_block = 0
        unassigned_in_block = 0
        
        start_r = (block_idx // 3) * 3
        start_c = (block_idx % 3) * 3
        
        for r_offset in range(3):
            for c_offset in range(3):
                r, c = start_r + r_offset, start_c + c_offset
                pos = (r, c)
                if pos in all_variables_set:
                    if pos in assignment:
                        mines_in_block += assignment[pos]
                    else:
                        unassigned_in_block += 1
                        
        if mines_in_block > 3:
            return False
        if mines_in_block + unassigned_in_block < 3:
            return False

    for (r, c), total in number_dicts.items():
        if not _neighbor_check(r, c, total, assignment, all_variables_set):
            return False
    return True


def _neighbor_check(r, c, total, assignment, all_variables_set):
    """
        Checks the number cell constraint for a given cell.
        Args:
            r (int): Row index of the number cell.
            c (int): Column index of the number cell.
            total (int): The number of mines that should be adjacent to the cell.
            assignment (dict): Current assignment of variables.
            all_variables_set (set): Set of all variable positions (row, col).
        Returns: True if the number cell constraint is satisfied, False otherwise.
    """
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    mine_count = 0
    unassigned_count = 0

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        pos = (nr, nc)
        
        if 0 <= nr < 9 and 0 <= nc < 9:
            if pos in all_variables_set: 
                if pos in assignment:
                    mine_count += assignment[pos]
                else:
                    unassigned_count += 1
    
    if mine_count > total:      # similar to previous checks; but make it the number cell specific
        return False
    if mine_count + unassigned_count < total:
        return False
        
    return True


def inference(var, variables, assigned, number_dicts):
    """
        Performs inference on the given variable using Sudoku and number cell constraints.
        Args:
            var (Variable): The variable to perform inference on.
            variables (dict): Dictionary mapping (row, col) to Variable objects.
            assigned (dict): Current assignment of variables.
            number_dicts (dict): Dictionary mapping (row, col) to integers for numbered cells.
        Returns: A tuple (success, logs) where success is a boolean indicating if inference was successful, and logs is a list of tuples (row, col, value) indicating pruned values.
    """

    (success_row, logs_row) = _sudo_inference(var, assigned, variables, 'row')
    if not success_row: return False, logs_row

    (success_col, logs_col) = _sudo_inference(var, assigned, variables, 'col')
    if not success_col: return False, logs_row + logs_col

    (success_block, logs_block) = _sudo_inference(var, assigned, variables, 'block')
    if not success_block: return False, logs_row + logs_col + logs_block

    (success_num, logs_num) = _num_constraint_inference(var, assigned, variables, number_dicts)
    if not success_num:
        return False, logs_row + logs_col + logs_block + logs_num

    all_logs = logs_row + logs_col + logs_block + logs_num
    return True, all_logs


def _sudo_inference(var: Variable, assigned, variables, op):
    """
        Performs Sudoku constraint inference on the given variable for a specific operation (row, col, block).
        Args:
            var (Variable): The variable to perform inference on.
            assigned (dict): Current assignment of variables.
            variables (dict): Dictionary mapping (row, col) to Variable objects.
            op (str): The operation to perform ('row', 'col', or 'block').
        Returns: A tuple (success, logs) where success is a boolean indicating if inference was successful, and logs is a list of tuples (row, col, value) indicating pruned values.
    """
    vr, vc = var.row, var.col
    logs = []
    mine = 0
    no_assign = []

    if op == 'row':
        # row check
        for c in range(9):
            coord = (vr, c)
            if coord in assigned:
                mine += assigned[coord]
            elif coord in variables:
                no_assign.append(variables[coord])

    elif op == 'col':
        for r in range(9):
            coord = (r, vc)
            if coord in assigned:
                mine += assigned[coord]
            elif coord in variables:
                no_assign.append(variables[coord])

    elif op == 'block':
        start_r, start_c = (vr // 3) * 3, (vc // 3) * 3

        for r_off in range(3):
            for c_off in range(3):
                coord = (start_r + r_off, start_c + c_off)

                if coord in assigned:
                    mine += assigned[coord]
                elif coord in variables:
                    no_assign.append(variables[coord])

    if mine == 3:
        for i in range(len(no_assign)):
            if 1 in no_assign[i].domain:
                no_assign[i].domain.remove(1)
                logs.append((no_assign[i].row, no_assign[i].col, 1))
            if len(no_assign[i].domain) == 0:
                return False, logs

    elif mine+len(no_assign) == 3:
        for i in range(len(no_assign)):
            if 0 in no_assign[i].domain:
                no_assign[i].domain.remove(0)
                logs.append((no_assign[i].row, no_assign[i].col, 0))
            if len(no_assign[i].domain) == 0:
                return False, logs

    return True, logs


def _num_constraint_inference(var: Variable, assigned, variables, number_dicts):
    """
        Performs number cell constraint inference on the given variable.
        Args:
            var (Variable): The variable to perform inference on.
            assigned (dict): Current assignment of variables.
            variables (dict): Dictionary mapping (row, col) to Variable objects.
            number_dicts (dict): Dictionary mapping (row, col) to integers for numbered cells.
        Returns: A tuple (success, logs) where success is a boolean indicating if inference was successful, and logs is a list of tuples (row, col, value) indicating pruned values.
    """
    logs = []
    vr, vc = var.row, var.col
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),          (0,  1),
                  (1,  -1), (1,  0), (1,  1)]

    for dr, dc in directions:
        nr, nc = vr + dr, vc + dc
        number_pos = (nr, nc)

        if number_pos in number_dicts:

            N = number_dicts[number_pos]
            mine = 0
            no_assign = []

            for dr_num, dc_num in directions:
                neighbor_r, neighbor_c = nr + dr_num, nc + dc_num
                neighbor_pos = (neighbor_r, neighbor_c)

                if neighbor_pos in assigned:
                    mine += assigned[neighbor_pos]
                elif neighbor_pos in variables:
                    no_assign.append(variables[neighbor_pos])

            if mine == N:
                for var_to_prune in no_assign:
                    if 1 in var_to_prune.domain:
                        var_to_prune.domain.remove(1)
                        logs.append((var_to_prune.row, var_to_prune.col, 1))
                        if not var_to_prune.domain:
                            return False, logs

            elif mine + len(no_assign) == N:
                for var_to_prune in no_assign:
                    if 0 in var_to_prune.domain:
                        var_to_prune.domain.remove(0)
                        logs.append((var_to_prune.row, var_to_prune.col, 0))
                        if not var_to_prune.domain:
                            return False, logs

    return True, logs


def is_complete(assigned, variables, numbers):
    """
        Checks if the assignment is complete.
        Args:
            assigned (dict): Current assignment of variables.
            variables (dict): Dictionary mapping (row, col) to Variable objects.
            numbers (dict): Dictionary mapping (row, col) to integers for numbered cells.
        Returns: True if the assignment is complete and valid, False otherwise.
    """
    return (len(assigned) == len(variables)
            and is_consistent(assigned, numbers, variables)
            and sum(assigned.values()) ==27)


total_nodes = 0 # total nodes generated
fin_lvl = 0     # solution's level


def backtrack(variables, numbers, assignments, lvl):
    """
        Backtracking search algorithm to solve the CSP.
        Args:
            variables (dict): Dictionary mapping (row, col) to Variable objects.
            numbers (dict): Dictionary mapping (row, col) to integers for numbered cells. 
            assignments (dict): Current assignment of variables.
            lvl (int): Current level of recursion.
        Returns: A complete assignment if a solution is found, None otherwise.
    """
    global total_nodes, fin_lvl
    total_nodes += 1

    if is_complete(assignments, variables, numbers):
        fin_lvl = lvl
        return assignments

    curr_var = select_unassigned_var(assignments, variables)

    all_vars_set = set(variables.keys())

    for choice in DOMAIN:
        if choice in curr_var.domain:
            assignments[(curr_var.row, curr_var.col)] = choice
            if is_consistent(assignments, numbers, all_vars_set):
                inf_res, logs = inference(curr_var, variables, assignments, numbers)
                if inf_res:
                    recur = backtrack(variables, numbers, assignments, lvl + 1)
                    if recur:
                        return recur
                # no matter inference success or recur fail, we need to rollback the pruned domains
                for r, c, v in logs:
                    variables[(r, c)].domain.add(v)
            del assignments[(curr_var.row, curr_var.col)]

    return None


def solver(ifile):
    """
        Main solver function to read input, build the constraint graph, and initiate backtracking search.
        Args:
            ifile (str): Path to the input file.
        Returns: A complete assignment if a solution is found, None otherwise.
    """
    variables, numbers = parse_input(ifile)
    assignments = {}   # format {(r, c): val, ...}
    build_constraint_graph(variables, numbers)
    return backtrack(variables, numbers, assignments, 0)


if __name__ == "__main__":
    """
        Usage: python xm2204_pl2999_solver.py <iPath> <oPath>
    """

    if len(sys.argv) != 3:
        print("Usage: python xm2204_pl2999_solver.py <iPath> <oPath>")
        sys.exit(1)

    ifile = sys.argv[1]
    ofile = sys.argv[2]
    os.makedirs(os.path.dirname(ofile), exist_ok=True)

    res = solver(ifile)

    if res is None:
        print(f"No solution found for {ifile}")
    else:
        table = [['0' for j in range(9)] for i in range(9)]
        for k, v in res.items():
            table[k[0]][k[1]] = str(v)

        buffer = ""
        buffer += f"{fin_lvl}\n"
        buffer += f"{total_nodes}\n"

        for row in table:
            _r = " ".join(row)
            buffer += f"{_r}\n"

        with open(ofile, "w") as f:
            f.write(buffer)

        print(f"Solution for {ifile} written to {ofile} (Level: {fin_lvl}, Nodes: {total_nodes})")