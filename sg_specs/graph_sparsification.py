from cmx import doc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sparse_graphs.asym_graph import AsymMesh
from sg_specs.utils import id2D, l2


def spec_dense_graph(xys, d_min):
    doc @ f"""
    Insert each vertex individually, and only when it is at least `d_min={d_min}` away 
    from existing nodes."""
    with doc:
        graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

        graph.extend(xys, images=xys, meta=xys)
        graph.update_zs()
        graph.update_edges()
    return graph


def spec_sparsify_graph(graph, xys, d_min):
    doc @ f"""
    ## In-place Sparsification

    We can run the following to sparsify the graph in-place.
    """
    with doc:
        graph.update_zs()  # Assume the embed_fn has been changed.
        graph.dedupe_(d_min=d_min)
        graph.update_edges()
    return graph


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        plt.arrow(x, y, (x_ - x) * 0.9, (y_ - y) * 0.8, **kwargs,
                  head_width=1, head_length=1, length_includes_head=True,
                  head_starts_at_zero=True, fc=color, ec=color)


if __name__ == '__main__':
    doc @ """
    # Sparsify An Existing Graph
    
    Most of the time it is easier to maintain a sparse graph and only add new vertices
    sparingly. However sometimes, it is nice to be able to sparsify a graph in-place.
    
    In this example we show how you can do that.
    """
    with doc:
        np.random.seed(100)
        xys = np.random.uniform(-20, 20, [1600, 2])

    dense_graph = spec_dense_graph(xys, d_min=2)
    dense_edges = [*dense_graph.edges][::10]

    sparse_graph = spec_sparsify_graph(dense_graph, xys, d_min=2)

    r = doc.table().figure_row()

    plt.gca().set_aspect('equal')
    for i, j in tqdm(dense_edges, desc="sparse"):
        a, b = dense_graph._meta['meta'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/dense_graph.png?ts={doc.now('%f')}", title="Dense (10% of the edges)")
    plt.close()

    plt.gca().set_aspect('equal')
    for i, j in tqdm(sparse_graph.edges, desc="sparsify"):
        a, b = sparse_graph._meta['meta'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/batch_graph.png?ts={doc.now('%f')}", title="Batch")
    plt.close()

    doc @ """
    """
    doc.flush()
