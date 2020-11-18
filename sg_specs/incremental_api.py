from cmx import doc
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from sparse_graphs.asym_graph import AsymMesh
from sg_specs.utils import id2D, l2

if __name__ == '__main__':
    doc @ """
    # Advanced Example
    """
    with doc:
        np.random.seed(100)

        d_min = 2
        xys = np.random.uniform(-20, 20, [200, 2])

    doc @ f"""
    Only insert nodes when it is at least `d_min={d_min}` away 
    from existing nodes.
    """
    with doc:
        graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)
        graph.extend(images=xys, meta=xys)
        graph.dedupe_(d_min)
        graph.update_edges()

    # plotting code
    nodes = np.stack(graph['meta']).T
    fig = plt.figure()
    plt.gca().set_aspect('equal')
    for i, j in tqdm(graph.edges, desc=" edges"):
        a, b = graph._meta['meta'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)

    doc @ """
    Now to continuously add more vertices to the graph
    """
    with doc:
        for i in trange(500, desc="extend"):
            new_xys = np.random.uniform(-20, 20, [200, 2])
            graph.sparse_extend(new_xys, d_min=d_min, meta=new_xys, )
            graph.update_edges()

    r = doc.table().figure_row()

    r.savefig(f"figures/high_level/before.png?ts={doc.now('%f')}", title="before", fig=fig)
    plt.close(fig)

    fig = plt.figure()
    nodes = np.stack(graph['meta']).T
    plt.gca().set_aspect('equal')
    for i, j in tqdm(graph.edges, desc=" edges"):
        a, b = graph._meta['meta'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/high_level/after.png?ts={doc.now('%f')}", title="after")
    plt.close()

    doc.flush()
