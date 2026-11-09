from loguru import logger

"""
Subcircuit helpers on the plain-list AIG representation from aig_loader.

Translation from the C++ classes:
    Gate*                -> int node index (AIGER variable = literal >> 1, as in to_gnn)
    gate->get_ch0/ch1()  -> and_children(g)[node]
    gate->get_level()    -> level[node]  (to_gnn(g)["level"])
    gate->get_input()    -> node is not an AND gate (not a key of and_children)
    Subcircuit           -> two lists: nodes, inputs
"""


def and_children(g: dict) -> dict[int, tuple[int, int]]:
    """AND node -> (child0, child1) node indices (inversion bits dropped)."""
    return {lhs >> 1: (a >> 1, b >> 1) for lhs, a, b in g["ands"]}

def _bfs_collect(start: int, depth: int, children, nodes, inputs, node_set, input_set):
    """Depth-limited BFS from `start` along child edges (start already collected).
    AND children become nodes, input/latch children become inputs."""
    layer = [start]
    for _ in range(depth):
        nxt = []
        for n in layer:
            for c in children.get(n, ()):
                if c == 0:                       # constant (C++: ch < 2)
                    continue
                if c not in children:            # C++: c->get_input()
                    if c not in input_set:
                        input_set.add(c); inputs.append(c)
                elif c not in node_set:
                    node_set.add(c); nodes.append(c); nxt.append(c)
        if not nxt:
            break
        layer = nxt

def ensure_parent_closure(g: dict, nodes: list[int], inputs: list[int], root: int,
                          level, children=None) -> tuple[list[int], list[int]]:
    """Add every AND gate (level <= root's level) whose two children are both
    in the subcircuit, repeatedly. Returns (nodes sorted by descending level, inputs).

    One pass in ascending level order reaches the same fixed point as the C++
    while(changed) loop: a gate's children always have a lower level, so they
    are final by the time the gate is visited."""
    children = children or and_children(g)
    nodes, inputs = list(nodes), list(inputs)
    node_set = set(nodes)
    in_sc = node_set | set(inputs)
    for p in sorted(children, key=lambda v: level[v]):
        if level[p] > level[root] or p in node_set:
            continue
        c0, c1 = children[p]
        if c0 == 0 or c1 == 0:               # constant child (C++: gate() returns null)
            continue
        if c0 in in_sc and c1 in in_sc:
            if p in in_sc:                   # was an input: promote to node
                inputs.remove(p)
            nodes.append(p); node_set.add(p); in_sc.add(p)
    nodes.sort(key=lambda v: -level[v])
    return nodes, inputs

def fix_missing_children(g: dict, nodes: list[int], inputs: list[int],
                         children=None) -> tuple[list[int], list[int]]:
    """Add every child of a subcircuit node that is in neither list to `inputs`,
    so the subcircuit's boundary is complete. Returns (nodes, inputs)."""
    children = children or and_children(g)
    inputs = list(inputs)
    seen = set(nodes) | set(inputs)
    for n in nodes:
        for c in children.get(n, ()):      # non-AND nodes have no children
            if c != 0 and c not in seen:   # 0 = constant (C++: gate() returns null)
                inputs.append(c)
                seen.add(c)
    return list(nodes), inputs

def gen_fsa_subcircuit(g: dict, root: int, level, fsa: set[int], fsa_inp: set[int],
                       children=None) -> tuple[list[int], list[int]]:
    """Subcircuit of FSA gates at or below `root`'s level, closed under parents.

    fsa / fsa_inp: node indices for which C++ is_fsa() / is_fsa_inp() is true.
    TODO: compute these; no equivalent exists in the list representation yet.
    Returns (nodes sorted by descending level, inputs)."""
    children = children or and_children(g)
    nodes, inputs = [], []
    for n in range(g["header"]["M"], -1, -1):    # C++: i = num_gates-1 .. 0
        if level[n] > level[root]:
            continue
        if n not in fsa and n not in fsa_inp:
            continue
        if n not in children or n in fsa_inp:    # primary input/latch, or FSA input
            inputs.append(n)
        else:
            nodes.append(n)
    # ensure_parent_closure already returns nodes sorted by descending level,
    return ensure_parent_closure(g, nodes, inputs, root, level, children)


def gen_faha_subcircuit(g: dict, ids: list[tuple[int, int]], faha: dict[int, set[tuple[int, int]]],
                        level, children=None) -> tuple[list[int], list[int]]:
    """Subcircuit of all gates belonging to any requested full/half-adder instance.

    ids:  requested (faha_id, type) pairs; pairs with id -1 are ignored.
    faha: node index -> set of (faha_id, type) the gate belongs to
          (C++ has_faha_id_with_type). A gate may belong to several instances.
          TODO: compute this; no equivalent exists in the list representation yet.
    Returns (nodes sorted by descending level, inputs)."""
    wanted = {(rid, rtype) for rid, rtype in ids if rid != -1}
    nodes = [n for n in range(g["header"]["M"], -1, -1)  # C++: i = num_gates-1 .. 0
             if faha.get(n, set()) & wanted]
    nodes, inputs = fix_missing_children(g, nodes, [], children)
    nodes.sort(key=lambda v: -level[v])
    return nodes, inputs


def extract_subcircuit_by_depth(g: dict, root: int, depth: int, level, dist, cone_id,
                                NN: int, children=None) -> tuple[list[int], list[int]]:
    """Subcircuit of gates within `depth` levels below `root`.

    dist, cone_id: per-node arrays for C++ get_dist() / get_cone_id().
    NN: the C++ constant of the same name.
    TODO: compute dist and cone_id; no equivalent exists in the list representation yet.
    Returns (nodes sorted by descending level, inputs)."""
    children = children or and_children(g)
    root_level = int(level[root])
    if root_level < depth:
        logger.warn(f"root level {root_level} is less than requested subcircuit "
                      f"depth {depth}, reset depth to {root_level}")
        depth = root_level

    nodes, inputs = [root], []
    node_set, input_set = {root}, set()
    _bfs_collect(root, depth, children, nodes, inputs, node_set, input_set)

    # any child of a collected node that is not collected becomes an input
    nodes, inputs = fix_missing_children(g, nodes, inputs, children)
    input_set = set(inputs)

    # Sparse subcircuit: seed more BFS runs from neighbouring cones of influence.
    # C++ computes root_level - NN in size_t, so it wraps around to a huge value
    # when NN > root_level; `root_level < NN` reproduces that.
    if dist[root] <= depth and (root_level < NN or len(nodes) < root_level - NN):
        root_coi = int(cone_id[root])
        M = g["header"]["M"]
        for neighbor_coi in range(root_coi - 1, -1, -1):
            if len(nodes) >= root_level:
                break
            candidates = [n for n in range(M, -1, -1)
                          if n in children and cone_id[n] >= neighbor_coi
                          and level[n] <= root_level]
            for cand in candidates:
                if cand in node_set:
                    continue
                node_set.add(cand);
                nodes.append(cand)
                _bfs_collect(cand, depth, children, nodes, inputs, node_set, input_set)
                if len(nodes) >= root_level:
                    break

    # add parents whose both children are present; also sorts by descending level
    return ensure_parent_closure(g, nodes, inputs, root, level, children)