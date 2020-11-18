from collections import defaultdict
from contextlib import contextmanager

import numpy as np

np.seterr(all='ignore')


class EdgeView:
    def __init__(self, pairwise, edge_mask, visited_mask, neighbor_r, neighbor_d_min=None, node_mask=None):
        self.pairwise = pairwise
        self.edge_mask = edge_mask
        self.visited_mask = visited_mask
        self.neighbor_r = neighbor_r
        self.neighbor_d_min = neighbor_d_min
        self.node_mask = node_mask

    def __len__(self):
        mask = self.edge_mask.copy()
        if self.neighbor_r is not None:
            mask &= self.pairwise < self.neighbor_r
        if self.neighbor_d_min is not None:
            mask &= self.pairwise > self.neighbor_d_min
        np.fill_diagonal(mask, 0)
        return mask.sum()

    def __iter__(self):
        mask = self.edge_mask.copy()
        if self.neighbor_r is not None:
            mask &= self.pairwise < self.neighbor_r
        if self.neighbor_d_min is not None:
            mask &= self.pairwise > self.neighbor_d_min
        for i, j in zip(*np.nonzero(mask)):
            if i == j:
                continue
            yield i, j

    def __getitem__(self, item):
        i, j = item
        ok = self.edge_mask[i, j]
        not_visited = self.visited_mask[i, j]
        return dict(weight=self.pairwise[i, j]) if ok and not_visited else {}

    def __call__(self, r=None, d_min=None):
        mask = self.edge_mask.copy()
        mask &= self.visited_mask.copy()
        if r is not None:
            mask &= self.pairwise < r
        if d_min is not None:
            mask &= self.pairwise > d_min

        return np.argwhere(mask)


# kernels
l2 = lambda z, z_prime=0: np.linalg.norm(z - z_prime, axis=-1, ord=2)
dot = lambda z, z_prime=0: z @ z_prime
cos = lambda z, z_prime=0: (z @ z_prime / np.linalg.norm(z) / np.linalg.norm(z_prime)).sum(axis=-1)


class GraphSummary:
    def __init__(self, graph):
        self.graph = graph

    @property
    def vertices(self):
        return len(self.graph)

    @property
    def edges(self):
        return len(self.graph.edges)

    @property
    def sparsity(self):
        return len(self.graph.edges) / len(self.graph.nodes)

    @property
    def islands(self):
        from graph_search.islands import islands
        return islands(self.graph)

    @property
    def pruned(self):
        return len(self.graph.pruned_edges)

    def __repr__(self):
        from textwrap import dedent

        graph = self.graph
        l = len(graph)
        return dedent(f"""
        {str(graph)}:
            neighbor r: {graph.neighbor_r}
            neighbor r (min): {graph.neighbor_d_min}
            vertices: {len(graph)}
            edges: {len(graph.edges)}
            sparsity: {len(graph.edges) / l if l else 'N/A'}
            islands: {self.islands}
            pruned: {self.pruned}
        """).strip()


def set_fig(aspect="equal"):
    import matplotlib.pyplot as plt
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    if aspect is not None:
        plt.gca().set_aspect(aspect)
    plt.tight_layout()


class MeshMemory:
    """First In, Density-Out Mesh Memory

    New memory are always written into the memory (writing logic can still
    occur outside by the consumer before it enters MM.add(). Old memory are
    pushed out via a priority weight that emulates the local density around
    that vertex. We remove vertices.
    """
    zs = None

    # we allow adding these two nodes to the graph for graph search.
    # z_start = None
    # z_goal = None

    def __init__(self, kernel_fn, n=100, latent_dim=10, neighbor_r=1, neighbor_d_min=None,
                 enable_soft_prune=True, embed_fn=None):
        """Mesh Memory

        :param kernel_fn: a python function for computing the distances between latent vectors
        :param n: n ≡ |V|, the cardinality of the graph.
        :param latent_dim: self-explanatory
        :param neighbor_r: self-explanatory
        :param neighbor_d_min: self-explanatory
        """
        self.n = n
        self.index = np.arange(n, dtype=int)
        self.neighbor_r = neighbor_r
        self.neighbor_d_min = neighbor_d_min
        self.latent_dim = latent_dim
        self.kernel_fn = kernel_fn or l2
        self.latent_dim = latent_dim
        self.zs = np.zeros([n, latent_dim])
        self.zs_mask = np.full(n, False)
        # use sparse graph implementation
        self.pairwise = np.full([n, n], float('inf'))
        self.edge_mask = np.full(self.pairwise.shape, True)
        self.visited_mask = np.full(self.pairwise.shape, True)
        self.weights = np.full(n, float('inf'))
        self.enable_soft_prune = enable_soft_prune
        self.embed_fn = embed_fn

    def __len__(self):
        return int(self.zs_mask.sum())

    def __repr__(self):
        l = len(self)
        s = f"n={self.n}, " if l < self.n else ""
        return f"{self.__class__.__name__}({l}, {s}latent_dim={self.latent_dim})"

    def update_zs(self):
        new_zs = self.embed_fn(self.images[self.zs_mask])
        self.add_zs(new_zs, spots=np.nonzero(self.zs_mask))

    def update_edges(self):
        raise RuntimeError('this is wrong')
        # as long as the indices do not change, updating the edges is okay.
        self.weights[:] = self.compute_weight(self.pairwise, self.neighbor_r, zs_mask=self.zs_mask)

    def update_r(self, neighbor_r, neighbor_d_min=None):
        self.neighbor_r = neighbor_r
        self.neighbor_d_min = neighbor_d_min
        self.weights[:] = self.compute_weight(self.pairwise, neighbor_r, zs_mask=self.zs_mask)

    def state_dict(self):
        return dict(zs=self.zs,
                    zs_mask=self.zs_mask,
                    edge_mask=self.edge_mask,
                    images=self.images,
                    meta=self.meta,
                    # save sparse indices
                    pairwise=self.pairwise[self.edge_mask],
                    visited_mask=self.visited_mask,
                    weights=self.weights, )

    def load_state_dict(self, d):
        for k, v in d:  # maintain reference.
            if k == "pairwise":
                continue
            getattr(self, k)[...] = v
            logger.print(f"{self}.{k} is loaded", color="green")

        self.pairwise[self.edge_mask] = d['pairwise']

    @property
    def pruned_edges(self):
        graph = self
        d = graph.neighbor_r
        pruned = (self.pairwise < d) & np.logical_not(graph.edge_mask) & \
                 graph.zs_mask[:, None] & graph.zs_mask[None, :]
        inds = np.argwhere(pruned)
        return inds

    @property
    def nodes(self):
        return self.index[self.zs_mask]

    @property
    def summary(self):
        return GraphSummary(self)

    # Methods for search
    def closest(self, v_or_z, r=None, d_min=None):
        vs = self.zs[v_or_z] if isinstance(v_or_z, int) else v_or_z
        ns, ds = self.query(vs, r=r, d_min=d_min)
        return ns

    def neighbors(self, n: int):
        assert isinstance(n, int) or isinstance(n, np.integer), "n needs to be an integer"
        # KISS implementation for the win!!
        row = self.pairwise[n]
        mask = self.visited_mask[n].copy()
        mask &= True if self.enable_soft_prune else self.edge_mask[n]
        # need to change this for soft pruning -- you still want the edges to be there.
        if self.neighbor_r:
            mask &= row < self.neighbor_r
        if self.neighbor_d_min:
            mask &= row > self.neighbor_d_min

        ds = row[mask]
        ord = np.argsort(ds)

        return self.index[:len(mask)][mask][ord]

    @property
    def edges(self):
        # info: always gets up-to-date r and d_min.
        # todo: add goal related logic.
        # edge_view current does not support using latent vectors to query the edge length.
        return EdgeView(self.pairwise, self.edge_mask, self.visited_mask, self.neighbor_r, self.neighbor_d_min)

    # todo: allow adding edges (make pairwise < threshold)
    def remove_similar(self, z, z_=None, *, r, add=False):
        m, _ = self.query([z], r)
        n = None if z_ is None else self.query([z_], r)[0]
        for u in m:
            self.edge_mask[u, n] = add

    def remove_incoming(self, z=None, *, r, img=None):
        """All incoming edges would be removed, but not out-going edges
        :param *:
        """
        if z is None:
            z, = self.embed_fn(img)

        m, _ = self.query(z[None, ...], r)
        for u in m:
            ns = self.neighbors(u)
            self.edge_mask[ns, u] = False
        return m

    def remove_toward(self, z, z_goal, r, r0, d0=0.2):
        z_delta = z_goal - z
        z_delta /= self.kernel_fn(z_delta)

        m, _ = self.query([z], r0)
        n, _ = self.query([z + z_delta], r)
        for u in m:
            for v in n:
                proj = np.clip((self.zs[v] - self.zs[u]) * z_delta, 0, float('inf'))
                if self.kernel_fn(proj) > d0:
                    self.edge_mask[u, v] = False

    def remove_edge(self, n, m, *, dilate=None):
        if dilate is not None:
            self.pairwise[n, m] *= dilate

        # still mark
        self.edge_mask[n, m] = False

    def mark(self, n, m):
        self.visited_mask[n, m] = False

    images = None
    meta = defaultdict(lambda: None)

    @property
    def all_images(self):
        """gives only the images being used."""
        return self.images[self.zs_mask]

    def add(self, zs=None, *, imgs, meta=None):
        if zs is None:
            zs = self.embed_fn(imgs)

        spots = self.add_zs(zs)
        if meta is not None:
            for ind, m in zip(spots, meta):
                self.meta[ind] = m

        if self.images is None:
            self.images = np.zeros((self.n,) + imgs.shape[1:])

        self.images[spots] = imgs

        return spots

        # n = len(zs)
        # # note: asymmetric
        # edges = np.concatenate([np.zeros([n, 1]), np.eye(n)[:, :-1]], axis=-1)

    @staticmethod
    def compute_weight(pairwise, neighbor_r, *, zs_mask=None):
        # note: we want more edges to have higher weights
        #  we want shorter distance to have higher weights
        #  ~
        #  We probably also want
        mask = pairwise < neighbor_r
        w = (1 / pairwise) * mask
        w = np.nansum(w, axis=-1)
        # note: what is the surface area?
        if zs_mask is not None:
            w[np.logical_not(zs_mask)] = float('inf')
        # self.logger.store_metrics(mask_sum=mask.sum(axis=-1), weights=w)
        # logger.log_metrics_summary()
        return w

    def add_zs(self, zs, spots=None):
        """complicated shit."""
        n = len(zs)
        # todo: make sure that weights are set to 0 upfront
        if spots is None:
            spots = np.argpartition(self.weights, -n)[-n:]
            spots_order = np.argsort(self.weights[spots])
            spots = spots[spots_order[::-1]]

        self.zs[spots] = zs
        self.zs_mask[spots] = True
        ds = self.kernel_fn(zs[:, None, :],
                            self.zs[self.zs_mask].reshape(-1, self.latent_dim)[None, :, :])
        rows = np.full([n, self.n], float('inf'))
        a, = np.nonzero(self.zs_mask)
        rows[:, a] = ds
        self.pairwise[spots] = rows
        self.pairwise.T[spots] = rows
        # note: easier to fill here.
        np.fill_diagonal(self.pairwise, float('inf'))
        # note: use rows filled with inf
        # Note-2: this is linear time.
        self.weights[spots] = self.compute_weight(self.pairwise[spots], self.neighbor_r)
        # self.weights = self.compute_weight(self.pairwise, self.neighbor_r, self.zs_mask)
        # debug: self.edge_mask[:] = False
        self.edge_mask[spots] = True
        self.edge_mask.T[spots] = True

        return spots

    def max_weight(self, no_inf=True):
        if no_inf:
            return self.weights[self.weights < float('inf')].max()
        return self.weights.max()

    def sparsify(self, r):
        all_nodes = set(self.nodes)
        while all_nodes:
            v = all_nodes.pop()
            ns, ds = self.query(self.zs[v], r)
            if ns is None:
                continue
            for n in ns:
                all_nodes.discard(n)
            ns = ns.tolist()
            ns.remove(v)
            self.remove_vertex(ns)

    def remove_vertex(self, ind):
        if ind is None:
            raise IndexError("None ind is not allowed since it removes everything.")
        self.zs_mask[ind] = False
        self.edge_mask[ind] = False
        self.edge_mask.T[ind] = False

    def remove_highest(self, n):
        spots = np.argsort(self.weights)[-n:]
        self.zs[spots] = 0
        self.zs_mask[spots] = False
        self.weights[spots] = float('inf')
        self.pairwise[spots] = float('inf')
        self.pairwise.T[spots] = float('inf')
        self.edge_mask[spots] = True
        self.edge_mask.T[spots] = True

    def pressure_release(self, threshold):
        # todo: iteratively prune instead.
        spots = self.weights > threshold
        self.zs[spots] = 0
        self.zs_mask[spots] = False
        self.weights[spots] = float('inf')
        self.pairwise[spots] = float('inf')
        self.pairwise.T[spots] = float('inf')
        self.edge_mask[spots] = True
        self.edge_mask.T[spots] = True

    def plot(self, filename='debug/mesh_memory/latent.png', title="Mesh", xlim=(-0.3, 0.3), ylim=(-0.3, 0.3),
             show=False, maxlen=None, thin=1):
        assert self.latent_dim == 2
        from ml_logger import logger
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(2.1, 2.4), dpi=70)
        if title is not None:
            plt.title(title)
        all_edges = list(self.edges)[:maxlen:thin]
        if show:
            all_edges = tqdm(all_edges, desc="all edges")
        for i, e in enumerate(all_edges):
            plt.plot(*self.zs[list(e)].T, color="#23aaff", alpha=0.2 + 0.8 * i / len(all_edges), linewidth=1)

        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        set_fig()
        logger.savefig(filename, dpi=300)

        if show:
            plt.gcf().set_facecolor("white")
            plt.show()
        plt.close()

    def plot_weights(self, max=30, bins=31):
        import matplotlib.pyplot as plt
        plt.hist(np.clip(self.weights[self.zs_mask], 0, max), bins=bins)
        plt.show()

    # Methods as a buffer
    def sample(self, size):
        return np.random.choice(self.index[self.zs_mask], size, replace=False)

    # Methods relevant to planning
    def localize(self, z, r=None, d_min=None):
        ns, ds = self.query(z, r=r, d_min=d_min)
        try:
            return ns[0], ds[0]
        except IndexError:
            return None, None

    @contextmanager
    def LocalizationContext(self, img, z=None, meta=None, suppressed=True):
        if not suppressed:
            assert self.zs_mask.sum() < (self.n - 1), "localization would overwrite existing vertices"
        elif self.zs_mask.sum() < (self.n - 1):
            self.remove_highest(1)
        if z is None:
            z, = self.embed_fn(img[None, ...])
        spot, = self.add(z[None, ...], imgs=img[None, ...], meta=meta)
        yield spot, z
        self.remove_vertex(spot)

    def query(self, zs, r=None, d_min=None):
        """query nodes with latent.

        we do not allow batch mode b/c speed up is negligible.
        """
        ds = self.kernel_fn(zs, self.zs)
        mask = np.full(ds.shape, True)
        if r is not None:
            mask &= ds < r
        if d_min is not None:
            mask &= ds > d_min

        selected_ds = ds[mask]
        selected_inds = self.index[mask]
        order = np.argsort(selected_ds)
        return selected_inds[order], selected_ds[order]

    def clear(self):
        # self.zs[:] = 0  # do not have to reset this one
        # self.images[:] = 0  # do not have to reset this one

        self.zs_mask[:] = False
        self.pairwise[:] = float('inf')
        self.edge_mask[:] = True
        self.visited_mask[:] = True
        self.weights[:] = float('inf')

    def plot_2d(self, filename=None, show=True, thin=1, fig=None,
                color="#e6e6e6", prune_color="red", vertex_color=None):
        import matplotlib.pyplot as plt
        from ml_logger import logger
        from tqdm import tqdm
        meta = self.meta

        if fig is None:
            plt.figure(figsize=(2.4, 2.1), dpi=120)

        for m, n in tqdm(list(self.edges)[::thin], desc="plot ℰ"):
            plt.plot([meta[m][0], meta[n][0]],
                     [meta[m][1], meta[n][1]],
                     '-', color=color, linewidth=1, alpha=0.3)

        if prune_color is not None:
            for m, n in self.pruned_edges:
                plt.plot([meta[m][0], meta[n][0]],
                         [meta[m][1], meta[n][1]],
                         '-', color=prune_color, linewidth=2, alpha=0.6)

        if vertex_color is not None:
            plt.plot(*np.array([meta[v] for v in self.nodes]).T, 'o', markersize=1, color=vertex_color)

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if filename:
            logger.savefig(filename, dpi=120)
        if show:
            plt.show()
            plt.close()

    def plot_3d(self):
        pass


if __name__ == '__main__':
    from tqdm import tqdm
    from ml_logger import logger


    class TestArgs:
        seed = 100
        latent_dim = 2
        plot_interval = None


    # the issue is that you are always removing edges from the graph.
    def make_transitions(*size, w=0.6, r=0.03):
        """
        :return: s, s'
        """
        s = (np.random.rand(*size) - 0.5) * w
        s_prime = s + (np.random.rand(*size) - 0.5) * r
        return s, s_prime


    def uniform_test():
        """Initial test with uniform samples"""
        s, s_prime = make_transitions(2000, TestArgs.latent_dim)
        trajs = np.array([s, s_prime]).transpose(1, 0, 2)

        mem = MeshMemory(None, n=1100, latent_dim=TestArgs.latent_dim, neighbor_r=0.035)

        for i, traj in enumerate(tqdm(trajs)):
            mem.add(traj)

            if TestArgs.plot_interval and i % TestArgs.plot_interval == 0:
                mem.plot(f"figures/mesh_{i:04d}.png")

        mem.plot(f"figures/mesh_all.png")
        print(f'{i}')


    def corner_transitions(*size, w=0.6, r=0.03):
        """
        :return: s, s'
        """
        s = np.random.rand(*size) * w
        s_prime = s + (np.random.rand(*size) - 0.5) * r
        return s, s_prime


    def dynamic_test(n=400, after=50, step=1, taper=100, show=False):
        from tqdm import trange
        np.random.seed(TestArgs.seed)

        mem = MeshMemory(None, n=10000, latent_dim=TestArgs.latent_dim,
                         neighbor_r=0.035, neighbor_d_min=0.02)

        for i in trange(after, n + taper + 1, step):
            s, s_prime = corner_transitions(5, TestArgs.latent_dim, w=0.3 * np.min([i / n, 1]))
            # trajs = np.array([s, s_prime]).transpose(1, 0, 2)
            mem.add_zs(s)
            # if i >= after and i % interval == 0:
            # mem.pressure_release(threshold=500)

            # mem.plot(f"figures/dynamic/mesh_{i:04d}.png", xlim=(0, 0.3), ylim=(0, 0.3),
            #          show=show, thin=1, maxlen=4000 * i // n)


    TestArgs.latent_dim = 32
    dynamic_test(show=False)

    # biased memory test
    # uniform memory test
    # uniform_test()
    # dynamic_test()
