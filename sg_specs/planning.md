
# Advanced Example

**Features**
- [x] test for distance to other nodes before insertion
- [x] run graph search with this graph and show the shortest path

```python
np.random.seed(100)
xys = np.random.uniform(-20, 20, [1600, 2])
```

## Sparse Graph Construction

Only insert nodes when it is at least `r_min=2` away 
from existing nodes.

```python
graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

for xy in xys:
    ds = graph.to_goal(zs_2=xy[None, :])
    # when the graph is empty, the ds.size can be zero.
    if ds.size == 0 or ds.min() >= r_min:
        graph.extend(xy[None, :], images=xy[None, :], meta=xy[None, :])

graph.update_zs()
graph.update_edges()
```

## Planning

This is useful during planning, where we need to insert current goal
to the graph.

```python
from graph_search import dijkstra

inds = graph.indices
np.random.shuffle(inds)
start, goal = inds[:2]
path, ds = dijkstra(graph, start, goal)
doc.print("path:", path)
doc.print("distances", ds)
```

```
path: [41, 139, 77, 214, 211]
distances [array([2.07942699]), array([2.44685013]), array([3.86278538]), array([3.84513508])]
```
| **Sparse** | **Plan** |
|:----------:|:--------:|
| <img style="align-self:center;" src="figures/before_planning.png?ts=862669" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/path.png?ts=264992" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |
