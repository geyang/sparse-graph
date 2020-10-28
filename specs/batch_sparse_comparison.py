from cmx import doc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sparse_graphs.asym_graph import AsymMesh, l2, id2D


def spec_individual_add(xys, r_min):
    doc @ f"""
    Insert each vertex individually, and only when it is at least `r_min={r_min}` away 
    from existing nodes."""
    with doc:
        graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

        for xy in xys:
            ds = graph.to_goal(zs_2=xy[None, :])
            # when the graph is empty, the ds.size can be zero.
            if ds.size == 0 or ds.min() >= r_min:
                graph.extend(xy[None, :], images=xy[None, :], meta=xy[None, :])

        graph.update_zs()
        graph.update_edges()
    return graph


def spec_batch_add(xys, r_min):
    doc @ f"""
    # Insert new vertices in a batched mode (slightly faster).
    
    We have to dedupe within the inserted batch, otherwise adding the whole batch is 
    the same as the non-sparse add. 
    """
    with doc:
        graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

        spots = graph.dedupe(images=xys, r_min=r_min)
        xys = xys[spots]
        ds = graph.to_goal(zs_2=xys)
        if ds.size == 0:
            graph.extend(xys, images=xys, meta=xys)
        else:
            m = ds.min(axis=-1) >= r_min
            if m.sum() > 0:
                graph.extend(xys[m], images=xys[m], meta=xys[m])
        graph.update_edges()
    return graph


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        plt.arrow(x, y, (x_ - x) * 0.9, (y_ - y) * 0.8, **kwargs,
                  head_width=1, head_length=1, length_includes_head=True,
                  head_starts_at_zero=True, fc=color, ec=color)


if __name__ == '__main__':
    doc @ """
    # Compare Adding New Vertices Individually vs In Batches
    """
    xys = np.random.uniform(-20, 20, [1600, 2])
    sparse_graph = spec_individual_add(xys, r_min=2)
    batch_graph = spec_batch_add(xys, r_min=2)

    r = doc.table().figure_row()

    nodes = np.stack(batch_graph.meta[batch_graph.z_mask]).T
    # plt.scatter(*nodes, color="#23aaff", edgecolor='none', alpha=0.6, s=50)
    plt.gca().set_aspect('equal')
    for i, j in tqdm(batch_graph.edges, desc="batched"):
        a, b = batch_graph.meta[[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/batch_graph.png?ts={doc.now('%f')}", title="Batch")
    plt.close()

    nodes = np.stack(sparse_graph.meta[sparse_graph.z_mask]).T
    # plt.scatter(*nodes, color="#23aaff", edgecolor='none', alpha=0.6, s=50)
    plt.gca().set_aspect('equal')
    for i, j in tqdm(sparse_graph.edges, desc="sparse"):
        a, b = sparse_graph.meta[[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/sparse_graph.png?ts={doc.now('%f')}", title="Sparse")
    plt.close()

    doc @ """
    When implemented correctly, the batched version gives identical result as 
    adding each vertex individually.
    """
    doc.flush()
