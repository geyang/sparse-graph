from cmx import doc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sparse_graphs.asym_graph import AsymMesh
from sg_specs.utils import id2D, l2


def spec_sparse_graph(xys, d_min):
    doc @ f"""
    ## Sparse Graph Construction

    Only insert nodes when it is at least `d_min={d_min}` away 
    from existing nodes.
    """
    with doc:
        graph = AsymMesh(n=10_000, k=6, dim=2, img_dim=[2], kernel_fn=l2, embed_fn=id2D, d_max=20)

        for xy in xys:
            ds = graph.to_goal(zs_2=xy[None, :])
            # when the graph is empty, the ds.size can be zero.
            if ds.size == 0 or ds.min() >= d_min:
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
    with doc:
        np.random.seed(100)
        xys = np.random.uniform(-20, 20, [1600, 2])

    graph = spec_sparse_graph(xys, d_min=2)

    doc @ """
    ## Planning
    
    This is useful during planning, where we need to insert current goal
    to the graph.
    """
    with doc:
        from graph_search import dijkstra

        inds = graph.indices
        np.random.shuffle(inds)
        start, goal = inds[:2]
        path, ds = dijkstra(graph, start, goal)
        doc.print("path:", path)
        doc.print("distances", ds)

    # Plot here.
    r = doc.table().figure_row()

    nodes = np.stack(graph['meta']).T
    # plt.scatter(*nodes, color="#23aaff", edgecolor='none', alpha=0.6, s=50)
    plt.gca().set_aspect('equal')
    for i, j in tqdm(graph.edges, desc="edges"):
        a, b = graph._meta['meta'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    r.savefig(f"figures/before_planning.png?ts={doc.now('%f')}", title="Sparse")
    plt.close()

    nodes = np.stack(graph['meta']).T
    plt.scatter(*nodes, color="black", edgecolor='none', alpha=0.1, s=50)
    plt.gca().set_aspect('equal')
    plot_trajectory_2d(graph._meta['meta'][path])

    r.savefig(f"figures/path.png?ts={doc.now('%f')}", title="Plan")
    plt.close()

    doc.flush()
