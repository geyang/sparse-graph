
# Sparsify An Existing Graph

Most of the time it is easier to maintain a sparse graph and only add new vertices
sparingly. However sometimes, it is nice to be able to sparsify a graph in-place.

In this example we show how you can do that.

```python
np.random.seed(100)
xys = np.random.uniform(-20, 20, [1600, 2])
```

Insert each vertex individually, and only when it is at least `d_min=2` away 
from existing nodes.
```python
graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

graph.extend(xys, images=xys, meta=xys)
graph.update_zs()
graph.update_edges()
```

## In-place Sparsification

We can run the following to sparsify the graph in-place.

```python
graph.update_zs()  # Assume the embed_fn has been changed.
graph.dedupe_(d_min=d_min)
graph.update_edges()
```
| **Dense (10% of the edges)** | **Batch** |
|:----------------------------:|:---------:|
| <img style="align-self:center;" src="figures/dense_graph.png?ts=295229" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center;" src="figures/batch_graph.png?ts=984425" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |


