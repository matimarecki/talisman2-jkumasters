from aig_loader import load_dir_gnn
from loguru import logger

data = load_dir_gnn("../examples")
# print(len(data))
# print(len(data))
for d in data:
    print(len(d["x"]))
d = data[6]
# print(d["x"])           # node features [N, 6]
# print(d["edge_index"])  # [2, E] connections, pointing from a gate's input to the gate
# print(d["edge_attr"])   # [E, 2] for each connection: inverted?, goes through a latch?
for key, value in d.items():
    print(key, value)
