
# Distribution of points

**to-dos**
- [x] test for distance to other nodes before insertion
- [x] run graph search with this graph and show shortest path

Now we have planning working. Can we remove vertices that are
too close to other vertices? Are we going to be able to preserve
the transitions? for GRED the relabeling still works.


Only insert nodes when it is at least `r_min=2` away from existing nodes.
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

## Adding and Removing Vertices

This is useful during planning, where we need to insert current goal
to the graph.

```python
from graph_search import dijkstra

xys = np.random.uniform(-20, 20, [1600, 2])
sparse_graph = spec_sparse_graph(xys, r_min=2)

inds = sparse_graph.indices
np.random.shuffle(inds)
start, goal = inds[:2]
path, ds = dijkstra(sparse_graph, start, goal)
```


Only insert nodes when it is at least `r_min=2` away from existing nodes.
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
| **Random** | **Sparse** | **Plan** |
|:----------:|:----------:|:--------:|
| <img style="align-self:center;" src="figures/random_graph.png?ts=658988" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/sparse_graph.png?ts=010004" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/path.png?ts=433629" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |
