from cmx import doc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sparse_graphs.asym_graph import AsymMesh, l2, id2D


def spec_random_graph(xys):
    graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=2)

    for _xys in np.split(xys, 10):
        graph.extend(_xys, images=_xys, meta=_xys)

    graph.update_zs()
    graph.update_edges()
    return graph


def spec_sparse_graph(xys, r_min):
    doc @ f"""
    Only insert nodes when it is at least `r_min={r_min}` away from existing nodes."""
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


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        plt.arrow(x, y, (x_ - x) * 0.9, (y_ - y) * 0.8, **kwargs,
                  head_width=1, head_length=1, length_includes_head=True,
                  head_starts_at_zero=True, fc=color, ec=color)


if __name__ == '__main__':
    doc @ """
    # Distribution of points
    
    **to-dos**
    - [x] test for distance to other nodes before insertion
    - [x] run graph search with this graph and show shortest path
    
    Now we have planning working. Can we remove vertices that are
    too close to other vertices? Are we going to be able to preserve
    the transitions? for GRED the relabeling still works.
    """
    xys = np.random.uniform(-20, 20, [1600, 2])
    random_graph = spec_random_graph(xys)
    sparse_graph = spec_sparse_graph(xys, r_min=2)

    doc @ """
    ## Adding and Removing Vertices
    
    This is useful during planning, where we need to insert current goal
    to the graph.
    """
    with doc:
        from graph_search import dijkstra

        xys = np.random.uniform(-20, 20, [1600, 2])
        sparse_graph = spec_sparse_graph(xys, r_min=2)

        inds = sparse_graph.indices
        np.random.shuffle(inds)
        start, goal = inds[:2]
        path, ds = dijkstra(sparse_graph, start, goal)

    r = doc.table().figure_row()

    nodes = np.stack(random_graph.meta[random_graph.z_mask]).T
    plt.scatter(*nodes, color="#23aaff", edgecolor='none', alpha=0.6, s=50)
    plt.gca().set_aspect('equal')
    for i, j in tqdm(random_graph.edges, desc="edges"):
        a, b = random_graph.meta[[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/random_graph.png?ts={doc.now('%f')}", title="Random")
    plt.close()

    nodes = np.stack(sparse_graph.meta[sparse_graph.z_mask]).T
    plt.scatter(*nodes, color="#23aaff", edgecolor='none', alpha=0.6, s=50)
    plt.gca().set_aspect('equal')
    for i, j in tqdm(sparse_graph.edges, desc="edges"):
        a, b = sparse_graph.meta[[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/sparse_graph.png?ts={doc.now('%f')}", title="Sparse")
    plt.close()

    nodes = np.stack(sparse_graph.meta[sparse_graph.z_mask]).T
    plt.scatter(*nodes, color="black", edgecolor='none', alpha=0.1, s=50)
    plt.gca().set_aspect('equal')
    plot_trajectory_2d(sparse_graph.meta[path])

    r.savefig(f"figures/path.png?ts={doc.now('%f')}", title="Plan")
    plt.close()

    doc.flush()
