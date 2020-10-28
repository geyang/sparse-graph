from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

id2D = lambda xy: xy[:, :2]
l2 = lambda a, b: np.linalg.norm(a - b, ord=2, axis=-1)


class EdgeView:
    def __init__(self, indices, adj, z_mask, ds, d_max):
        self.indices = indices
        self.adj = adj
        self.z_mask = z_mask
        self._ds = ds
        self.d_max = d_max

    def __len__(self):
        return (self._ds < self.d_max).sum()

    def __iter__(self):
        for i, js, ds in zip(self.indices, self.adj[self.z_mask], self._ds[self.z_mask]):
            for j, d in zip(js, ds):
                if d < self.d_max:
                    yield i, j

    def __getitem__(self, item):
        i, j = item
        try:
            ind = np.where(self.adj[i] == j)[0]
            return {"weight": self._ds[i, ind]}
        except:
            return {"weight": float('inf')}
        # ok = self.edge_mask[i, j]
        # not_visited = self.visited_mask[i, j]
        # return dict(weight=self.pairwise[i, j]) if ok and not_visited else {}

    @property
    def ds(self):
        return self._ds[self._ds < self.d_max]


class AsymMesh:
    zs = None
    zs_2 = None
    images = None
    meta = None
    z_mask = None

    def __init__(self, n, dim, k, kernel_fn, embed_fn=None,  img_dim=None, neighbor_r=1, d_max=1):
        self.n = n
        self.dim = dim
        self.k = k
        self.kernel_fn = kernel_fn
        self.embed_fn = embed_fn
        self.img_dim = img_dim
        self.neighbor_r = neighbor_r
        self.d_max = d_max

        if img_dim is not None:
            self.images = np.zeros([n, *img_dim])
        self.zs = np.zeros([n, dim])
        self.zs_2 = np.zeros([n, dim])
        self.z_mask = np.full(n, False)
        self.ds = np.full([n, k], float('inf'))
        self.adj = np.full([n, k], np.nan, dtype=np.int32)
        self.meta = np.empty([n], dtype=object)

    def __len__(self):
        return self.z_mask.sum()

    def expand(self, new_n):
        """expand the size of the graph"""
        pass

    def extend(self, zs=None, images=None, meta=None):
        if zs is None:
            assert images is not None, "images has to be specified if zs is None."
            zs = self.embed_fn(images)
        l, dim = zs.shape
        spots = np.argpartition(self.z_mask, l)[:l]
        self.zs[spots] = zs
        if images is not None:
            self.images[spots] = images
        if meta is not None:
            for s, m in zip(spots, meta):
                self.meta[s] = deepcopy(m)  # avoid deletion by numpy GC
        self.z_mask[spots] = True
        return spots

    @property
    def indices(self):
        return np.arange(self.n, dtype=int)[self.z_mask]

    def to_goal(self, zs_2=None, images=None):
        if zs_2 is None:
            assert images is not None, "images has to be specified if zs is None."
            zs_2 = self.embed_fn(images)
        zs = self.zs[self.z_mask]
        pairwise = self.kernel_fn(zs[:, None, :], zs_2[None, ...])
        mask = pairwise >= self.d_max
        pairwise[mask] = float('inf')
        return pairwise

    def dedupe(self, zs=None, images=None, *, r_min):
        if zs is None:
            assert images is not None, "Need `images` when `zs` is None."
            zs = self.embed_fn(images)
        pairwise = self.kernel_fn(zs[:, None, :], zs[None, ...])
        pairwise[np.eye(len(zs), dtype=bool)] = float('inf')
        spots = []
        for row_n, row in enumerate(pairwise):
            if row_n and row[:row_n].min() < r_min:
                pairwise[:, row_n] = float('inf')
            else:
                spots.append(row_n)
        return spots

    def dedupe_(self, r_min):
        mask_spots = self.dedupe(self.zs[self.z_mask], r_min=r_min)
        spots = self.indices[mask_spots]
        self.z_mask[:] = False
        self.z_mask[spots] = True

    def update_zs(self):
        self.zs[self.z_mask] = self.embed_fn(self.images[self.z_mask])

    def update_edges(self):
        zs, l = self.zs[self.z_mask], self.z_mask.sum()
        # zs_2 = self.zs_2[self.z_mask]
        pairwise = self.kernel_fn(zs[:, None, :], zs[None, ...])
        mask = pairwise >= self.d_max
        pairwise[mask] = float('inf')
        pairwise[np.eye(l, dtype=bool)] = float('inf')
        knn = np.argpartition(pairwise, self.k, axis=-1)[:, :self.k]
        indices = self.indices
        for ind, ds, nn, m in zip(indices, pairwise, knn, mask):
            self.ds[ind] = ds[nn]
            self.adj[ind, :] = indices[nn]

    @property
    def edges(self):
        return EdgeView(indices=self.indices, adj=self.adj, z_mask=self.z_mask, ds=self.ds, d_max=self.d_max)

    def neighbors(self, n):
        return [n for n, d in zip(self.adj[n], self.ds[n]) if d < self.d_max]

    # plotting and vis
    def plot_2d(self, filename=None, title=None, **_):
        from ml_logger import logger

        logger.print(f"n: {len(self)} e: {len(self.edges)} d/mean: {self.ds.mean()}", file="graph_summary.md")
        # edges = list(self.edges)[::ceil(len(self.edges) / 4000)]
        edges = self.edges
        for i, j in tqdm(edges, desc="painting edges"):
            a, b = self.meta[[i, j]]
            plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
        plt.gca().set_aspect('equal')
        if title:
            plt.title(title)
        plt.tight_layout()
        if filename:
            logger.savefig(filename, **_)
            plt.close()

    def plot_2d_scatter(self, filename=None, title=None, **_):
        from ml_logger import logger

        xys = np.stack(self.meta[self.indices])

        plt.scatter(*xys.T, color="gray", s=40)
        plt.gca().set_aspect('equal')
        if title:
            plt.title(title)
        plt.tight_layout()
        if filename:
            logger.savefig(filename, **_)
            plt.close()
