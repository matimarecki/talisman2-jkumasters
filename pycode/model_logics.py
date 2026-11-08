"""
Train a GNN to imitate select_subgraph(g, highlighted_node) on AIGs.
One graph per step, no batching.

    python gnn_example.py DIR                  # train on .aig/.aag files in DIR
    python gnn_example.py DIR FILE NODE        # then predict for NODE in FILE
"""
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from aig_loader import load_dir, load_file, to_gnn
from loguru import logger


def select_subgraph(g: dict, node: int) -> set[int]:
    # todo: create python implementation for select_subgraph c++ function in talisman2
    """YOUR ALGORITHM HERE. Node numbering as in to_gnn (variable v = literal >> 1;
    outputs follow as M+1, M+2, ...). Placeholder: cone of influence."""
    M = g["header"]["M"]
    fanin = {lhs >> 1: (a >> 1, b >> 1) for lhs, a, b in g["ands"]}
    fanin |= {lit >> 1: (nxt >> 1,) for lit, nxt, _ in g["latches"]}
    fanin |= {M + 1 + i: (lit >> 1,) for i, lit in enumerate(g["outputs"] + g["bad"])}
    seen, stack = {node}, [node]
    while stack:
        for u in fanin.get(stack.pop(), ()):
            if u not in seen:
                seen.add(u); stack.append(u)
    return seen - {0}


def example(g: dict, node: int) -> dict:
    """GNN input tensors for graph g with `node` highlighted, plus labels."""
    d = to_gnn(g, highlight=node)
    y = torch.zeros(d["num_nodes"])
    y[list(select_subgraph(g, node))] = 1.0
    return {"x": torch.from_numpy(d["x"]), "edge_index": torch.from_numpy(d["edge_index"]),
            "edge_attr": torch.from_numpy(d["edge_attr"]), "highlight": node, "y": y}


class Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fwd, self.bwd = nn.Linear(h + 2, h), nn.Linear(h + 2, h)
        self.upd = nn.GRUCell(2 * h, h)

    def forward(self, h, ei, ea):
        src, dst = ei
        m_f = torch.zeros_like(h).index_add_(0, dst, self.fwd(torch.cat([h[src], ea], 1)))
        m_b = torch.zeros_like(h).index_add_(0, src, self.bwd(torch.cat([h[dst], ea], 1)))
        return self.upd(torch.relu(torch.cat([m_f, m_b], 1)), h)


class Selector(nn.Module):
    def __init__(self, in_dim=7, h=64, layers=10):
        super().__init__()
        self.embed = nn.Linear(in_dim, h)
        self.layers = nn.ModuleList(Layer(h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, e):
        h = self.embed(e["x"])
        for layer in self.layers:
            h = layer(h, e["edge_index"], e["edge_attr"])
        h_hl = h[e["highlight"]].expand_as(h)       # every node also sees the highlighted node
        return self.head(torch.cat([h, h_hl], 1)).squeeze(-1)  # one logit per node


def make_examples(graphs, per_graph=8):
    out = []
    for g in graphs:
        nodes = [v for v, t in enumerate(to_gnn(g)["node_type"]) if t not in (0, 1)]  # not const/input
        out += [example(g, v) for v in random.sample(nodes, min(per_graph, len(nodes)))]
    return out


def train(graphs, epochs=20, model: Selector=Selector()):
    random.shuffle(graphs)
    k = max(1, len(graphs) // 10)
    train_set, test_set = make_examples(graphs[k:]), make_examples(graphs[:k])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))  # selected nodes are the minority
    for epoch in range(1, epochs + 1):
        st = time.perf_counter()
        random.shuffle(train_set)
        for e in train_set:
            loss = lossf(model(e), e["y"])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            exact = np.mean([((model(e) > 0).float() == e["y"]).all().item() for e in test_set])
        logger.info(f"epoch {epoch}  loss {loss.item():.3f}  test: exact subgraph {exact:.3f}")
        logger.info(f"epoch {epoch} finished in {time.perf_counter()-st:.4f}s")
    # return model


def predict(model, g, node):
    e = example(g, node)
    with torch.no_grad():
        pred = set(torch.nonzero(model(e) > 0).flatten().tolist())
    logger.info(f"GNN selected:    {sorted(pred)}\nselect_subgraph: {sorted(torch.nonzero(e['y']).flatten().tolist())}")


if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    # model = train(load_dir("../examples"),epochs=5)
    graphlist = load_dir("../examples")
    model = Selector(in_dim=7, h=32, layers=4)
    train(graphs=graphlist, epochs=50, model=model)

    # predict(model, load_file("../examples/abc4.aig"), 2)
    torch.save(model.state_dict(), "models/aig_gnn_basics_50epochs.pt")
    predict(model, load_file("../examples/abc4.aig"), 2)

    train(graphs=graphlist, epochs=50, model=model)


    torch.save(model.state_dict(), "models/aig_gnn_basics_100epochs.pt")
    predict(model, load_file("../examples/abc4.aig"), 2)
    # if len(sys.argv) == 4:
    #     predict(model, load_file(sys.argv[2]), int(sys.argv[3]))