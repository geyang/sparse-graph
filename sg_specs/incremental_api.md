
# Advanced Example

```python
np.random.seed(100)

r_min = 2
xys = np.random.uniform(-20, 20, [200, 2])
```

Only insert nodes when it is at least `r_min=2` away 
from existing nodes.

```python
graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)
graph.extend(images=xys, meta=xys)
graph.dedupe_(r_min)
graph.update_edges()
```

Now to continuously add more vertices to the graph

```python
for i in trange(500, desc="extend"):
    new_xys = np.random.uniform(-20, 20, [200, 2])
    graph.sparse_extend(new_xys, r_min=r_min, meta=new_xys)
    graph.update_edges()
```
| **before** | **after** |
|:----------:|:---------:|
| <img style="align-self:center;" src="figures/high_level/before.png?ts=133434" image="None" styles="{'margin': '0.5em'}" width="None" height="None" fig="Figure(640x480)"/> | <img style="align-self:center;" src="figures/high_level/after.png?ts=826896" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> |

