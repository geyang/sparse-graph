
# Compare Adding New Vertices Individually vs In Batches

```python
np.random.seed(100)
xys = np.random.uniform(-20, 20, [1600, 2])
```

Insert each vertex individually, and only when it is at least `r_min=2` away 
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

# Insert new vertices in a batched mode (slightly faster).

```python
graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)
```

We have to dedupe within the inserted batch, otherwise adding the whole batch is 
the same as the non-sparse add. 

```python
spots = graph.dedupe(images=xys, r_min=r_min)
xys = xys[spots]
ds = graph.to_goal(zs_2=xys)
if ds.size == 0:
    graph.extend(xys, images=xys, meta=xys)
else:
    m = ds.min(axis=-1) >= r_min
    if m.any():
        graph.extend(xys[m], images=xys[m], meta=xys[m])
graph.update_edges()
```
| **Batch** | **Incremental Sparsification** |
|:---------:|:------------------------------:|
| <img style="align-self:center;" src="figures/batch_graph.png?ts=477817" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/sparse_graph.png?ts=933042" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |


When implemented correctly, the batched version gives identical result as 
adding each vertex individually.
