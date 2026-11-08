#!/usr/bin/env python3
"""
Load AIGER and-inverter graphs (.aig binary, .aag ASCII) into plain Python lists.

Literal convention (AIGER): variable v has literal 2*v; literal 2*v+1 is its
negation. Literal 0 = constant FALSE, 1 = constant TRUE.

Each graph becomes a dict:
    {
      "file":    path,
      "header":  {"M":..,"I":..,"L":..,"O":..,"A":..,"B":..,"C":..,"J":..,"F":..},
      "inputs":  [lit, ...],
      "latches": [(lit, next_lit, init), ...],  # init: 0, 1, or lit (= uninitialized)
      "outputs": [lit, ...],
      "bad":     [lit, ...],
      "constraints": [lit, ...],
      "justice": [[lit, ...], ...],
      "fairness": [lit, ...],
      "ands":    [(lhs, rhs0, rhs1), ...],      # lhs = rhs0 AND rhs1
      "symbols": {"i": {idx: name}, "l": {...}, "o": {...}, ...},
      "comments": [str, ...],
    }

Usage:
    python aig_loader.py DIR            # loads every *.aig / *.aag in DIR
    from aig_loader import load_dir; graphs = load_dir("DIR")

For graph neural networks:
    from aig_loader import load_dir_gnn, to_pyg
    data = load_dir_gnn("DIR")                       # numpy arrays (x, edge_index, edge_attr, ...)
    pyg  = [to_pyg(g) for g in load_dir("DIR")]      # torch_geometric Data objects
"""
import sys
from pathlib import Path


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def line(self) -> str:
        end = self.data.find(b"\n", self.pos)
        if end < 0:
            end = len(self.data)
        s = self.data[self.pos:end].decode("ascii", errors="replace").rstrip("\r")
        self.pos = end + 1
        return s

    def ints(self) -> list[int]:
        return [int(t) for t in self.line().split()]

    def varint(self) -> int:
        x, shift = 0, 0
        while True:
            if self.pos >= len(self.data):
                raise ValueError("unexpected EOF in binary AND section")
            b = self.data[self.pos]
            self.pos += 1
            x |= (b & 0x7F) << shift
            if not b & 0x80:
                return x
            shift += 7

    def eof(self) -> bool:
        return self.pos >= len(self.data)


def parse_aiger(data: bytes, name: str = "<bytes>") -> dict:
    r = _Reader(data)
    head = r.line().split()
    if not head or head[0] not in ("aig", "aag"):
        raise ValueError(f"{name}: not an AIGER file (header {head[:1]})")
    binary = head[0] == "aig"
    nums = [int(x) for x in head[1:]]
    if len(nums) < 5:
        raise ValueError(f"{name}: header needs at least M I L O A")
    nums += [0] * (9 - len(nums))
    M, I, L, O, A, B, C, J, F = nums[:9]

    # Inputs
    if binary:
        inputs = [2 * (i + 1) for i in range(I)]
    else:
        inputs = [r.ints()[0] for _ in range(I)]

    # Latches
    latches = []
    for i in range(L):
        t = r.ints()
        if binary:
            lhs, nxt, rest = 2 * (I + i + 1), t[0], t[1:]
        else:
            lhs, nxt, rest = t[0], t[1], t[2:]
        init = rest[0] if rest else 0
        latches.append((lhs, nxt, init))

    outputs = [r.ints()[0] for _ in range(O)]
    bad = [r.ints()[0] for _ in range(B)]
    constraints = [r.ints()[0] for _ in range(C)]
    jsizes = [r.ints()[0] for _ in range(J)]
    justice = [[r.ints()[0] for _ in range(n)] for n in jsizes]
    fairness = [r.ints()[0] for _ in range(F)]

    # AND gates
    ands = []
    if binary:
        for i in range(A):
            lhs = 2 * (I + L + i + 1)
            rhs0 = lhs - r.varint()
            rhs1 = rhs0 - r.varint()
            ands.append((lhs, rhs0, rhs1))
    else:
        for _ in range(A):
            lhs, rhs0, rhs1 = r.ints()
            ands.append((lhs, rhs0, rhs1))

    # Symbol table and comments
    symbols = {k: {} for k in "ilobcjf"}
    comments = []
    while not r.eof():
        ln = r.line()
        if ln == "c":
            rest = r.data[r.pos:].decode("utf-8", errors="replace")
            comments = rest.splitlines()
            break
        if not ln:
            continue
        kind = ln[0]
        if kind in symbols:
            idx, _, sym = ln[1:].partition(" ")
            if idx.isdigit():
                symbols[kind][int(idx)] = sym

    return {
        "file": name,
        "header": dict(zip("MILOABCJF", (M, I, L, O, A, B, C, J, F))),
        "inputs": inputs,
        "latches": latches,
        "outputs": outputs,
        "bad": bad,
        "constraints": constraints,
        "justice": justice,
        "fairness": fairness,
        "ands": ands,
        "symbols": symbols,
        "comments": comments,
    }


def load_file(path) -> dict:
    path = Path(path)
    return parse_aiger(path.read_bytes(), str(path))


def load_dir(directory, patterns=("*.aig", "*.aag")) -> list[dict]:
    """Return a list of parsed graphs, one per AIGER file in `directory`."""
    d = Path(directory)
    files = sorted({p for pat in patterns for p in d.glob(pat)})
    graphs = []
    for p in files:
        try:
            graphs.append(load_file(p))
        except Exception as e:  # keep going on a bad file
            print(f"skipping {p}: {e}", file=sys.stderr)
    return graphs


# ---------------------------------------------------------------------------
# GNN-ready conversion
# ---------------------------------------------------------------------------
# Node types (one-hot columns of the feature matrix)
NODE_TYPES = ("const", "input", "latch", "and", "output")
CONST, INPUT, LATCH, AND, OUTPUT = range(len(NODE_TYPES))
# Edge attribute columns
EDGE_ATTRS = ("inverted", "sequential")


def to_gnn(g: dict, include_bad: bool = True, highlight=None) -> dict:
    """
    Convert a parsed AIG into arrays for graph neural networks.

    Nodes:  index v (0..M) is AIGER variable v (0 = constant FALSE), followed
            by one explicit node per primary output (and per bad-state
            property if include_bad). Inverters are NOT nodes; they are edge
            attributes, which is the usual choice for AIG learning.
    Edges:  directed fanin -> fanout, so messages follow signal flow.
            AND gates get 2 incoming edges, outputs 1, latches 1 (from their
            next-state function, marked sequential=1 since it closes a cycle).

    highlight: optional node index or list of node indices to mark as the
            query target (e.g. the node of a safety property: "can this
            signal ever be 1?"). Sets the last feature column to 1 for those
            nodes so the GNN knows which vertex the question is about.

    Returns a dict of numpy arrays:
        x           float32 [N, 7]   one-hot node type (NODE_TYPES), normalized
                                     logic level, highlight flag
        node_type   int64   [N]      index into NODE_TYPES
        level       int64   [N]      logic depth (inputs/latches/const = 0)
        edge_index  int64   [2, E]   row 0 = source, row 1 = target
        edge_attr   float32 [E, 2]   (EDGE_ATTRS) inverted, sequential
        output_nodes int64  [O(+B)]  node indices of the output nodes
        num_nodes   int
    """
    import numpy as np

    M = g["header"]["M"]
    roots = list(g["outputs"]) + (list(g["bad"]) if include_bad else [])
    N = M + 1 + len(roots)

    node_type = np.full(N, -1, dtype=np.int64)
    node_type[0] = CONST
    for lit in g["inputs"]:
        node_type[lit >> 1] = INPUT
    for lit, _, _ in g["latches"]:
        node_type[lit >> 1] = LATCH
    for lhs, _, _ in g["ands"]:
        node_type[lhs >> 1] = AND
    out_nodes = np.arange(M + 1, N, dtype=np.int64)
    node_type[out_nodes] = OUTPUT
    # Variables declared in M but never defined: treat as constants
    node_type[node_type < 0] = CONST

    src, dst, inv, seq = [], [], [], []

    def edge(lit, target, sequential=0):
        src.append(lit >> 1); dst.append(target)
        inv.append(lit & 1); seq.append(sequential)

    for lhs, a, b in g["ands"]:
        edge(a, lhs >> 1)
        edge(b, lhs >> 1)
    for lit, nxt, _ in g["latches"]:
        edge(nxt, lit >> 1, sequential=1)
    for node, lit in zip(out_nodes, roots):
        edge(lit, int(node))

    # Logic level. ASCII files need not be topologically sorted, so use a
    # memoized iterative DFS over the AND fanins.
    fanin = {lhs >> 1: (a >> 1, b >> 1) for lhs, a, b in g["ands"]}
    level = np.zeros(N, dtype=np.int64)
    done = np.zeros(N, dtype=bool)
    for start in fanin:
        stack = [start]
        while stack:
            v = stack[-1]
            if done[v] or v not in fanin:
                done[v] = True; stack.pop(); continue
            pending = [u for u in fanin[v] if not done[u]]
            if pending:
                stack.extend(pending)
            else:
                level[v] = 1 + max(level[u] for u in fanin[v])
                done[v] = True; stack.pop()
    for node, lit in zip(out_nodes, roots):
        level[node] = level[lit >> 1] + 1

    hl = np.zeros(N, dtype=bool)
    if highlight is not None:
        idx = np.atleast_1d(np.asarray(highlight, dtype=np.int64))
        if idx.size and (idx.min() < 0 or idx.max() >= N):
            raise IndexError(f"highlight node out of range 0..{N - 1}")
        hl[idx] = True

    x = np.zeros((N, len(NODE_TYPES) + 2), dtype=np.float32)
    x[np.arange(N), node_type] = 1.0
    x[:, -2] = level / max(1, level.max())
    x[:, -1] = hl

    return {
        "highlight": np.flatnonzero(hl),
        "x": x,
        "node_type": node_type,
        "level": level,
        "edge_index": np.array([src, dst], dtype=np.int64).reshape(2, -1),
        "edge_attr": np.array([inv, seq], dtype=np.float32).T.reshape(-1, 2),
        "output_nodes": out_nodes,
        "num_nodes": N,
    }


def to_pyg(g: dict, **kw):
    """Same as to_gnn, wrapped as a torch_geometric.data.Data object."""
    import torch
    from torch_geometric.data import Data
    d = to_gnn(g, **kw)
    return Data(
        x=torch.from_numpy(d["x"]),
        edge_index=torch.from_numpy(d["edge_index"]),
        edge_attr=torch.from_numpy(d["edge_attr"]),
        node_type=torch.from_numpy(d["node_type"]),
        level=torch.from_numpy(d["level"]),
        output_nodes=torch.from_numpy(d["output_nodes"]),
        highlight=torch.from_numpy(d["highlight"]),
        num_nodes=d["num_nodes"],
        name=g["file"],
    )


def load_dir_gnn(directory, **kw) -> list[dict]:
    """Load every AIGER file in `directory` straight into GNN-ready arrays."""
    return [to_gnn(g, **kw) | {"file": g["file"]} for g in load_dir(directory)]


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    graphs = load_dir(directory)
    for g in graphs:
        h = g["header"]
        print(f"{g['file']}: inputs={h['I']} latches={h['L']} "
              f"outputs={h['O']} ands={h['A']}")
    print(f"{len(graphs)} graph(s) loaded")