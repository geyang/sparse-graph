from collections import defaultdict
from copy import deepcopy
import numpy as np


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
    _zs = None
    _zs_2 = None
    _images = None
    _meta = None
    z_mask = None

    def __init__(self, n, dim, k, kernel_fn, embed_fn=None, img_dim=None, d_max=1):
        self.n = n
        self.dim = dim
        self.k = k
        self.kernel_fn = kernel_fn
        self.embed_fn = embed_fn
        self.img_dim = img_dim
        self.d_max = d_max

        self._zs = np.zeros([n, dim])
        self._zs_2 = np.zeros([n, dim])
        self.z_mask = np.full(n, False)
        self.ds = np.full([n, k], float('inf'))
        self.adj = np.full([n, k], np.nan, dtype=np.int32)
        self._meta = defaultdict(lambda: np.empty([n], dtype=object))

    def keys(self):
        return self._meta.keys()

    def items(self):
        yield from ([k, self[k]] for k in self.keys())

    def __len__(self):
        return self.z_mask.sum()

    def expand(self, new_n):
        """expand the size of the graph"""
        pass

    def extend(self, zs=None, images=None, **meta):
        if zs is None:
            assert images is not None, "images has to be specified if zs is None."
            zs = self.embed_fn(images)
        l, dim = zs.shape
        spots = np.argpartition(self.z_mask, l)[:l]
        self._zs[spots] = zs
        if images is not None:
            self._meta['images'][spots] = list(images)
        for k, v in meta.items():
            self._meta[k][spots] = list(v)

        self.z_mask[spots] = True
        return spots

    @property
    def zs(self):
        return self._zs[self.z_mask]

    @property
    def zs_2(self):
        return self._zs_2[self.z_mask]

    @property
    def indices(self):
        return np.arange(self.n, dtype=int)[self.z_mask]

    @property
    def pairwise(self):
        zs = self._zs[self.z_mask]
        return self.kernel_fn(zs[:, None, :], zs[None, ...])

    def __getitem__(self, item):
        if isinstance(item, str):
            value = self._meta.get(item, None)
            return None if value is None else value[self.z_mask]

    @property
    def d0(self):
        l = len(self)
        inverse_index = np.full(self.n, -1, dtype=int)
        inverse_index[self.indices] = range(l)
        pairwise = np.full([l, l], float('inf'))
        np.fill_diagonal(pairwise, 0)
        for n, i in enumerate(self.indices):
            pairwise[n, inverse_index[self.adj[i]]] = self.ds[i]
        return pairwise

    def to_goal(self, zs_2=None, images=None):
        if zs_2 is None:
            assert images is not None, "images has to be specified if zs is None."
            zs_2 = self.embed_fn(images)
        zs = self._zs[self.z_mask]
        pairwise = self.kernel_fn(zs[:, None, :], zs_2[None, ...])
        mask = pairwise >= self.d_max
        pairwise[mask] = float('inf')
        return pairwise

    def dedupe(self, zs=None, images=None, *, d_min):
        if zs is None:
            assert images is not None, "Need `images` when `zs` is None."
            zs = self.embed_fn(images)
        pairwise = self.kernel_fn(zs[:, None, :], zs[None, ...])
        pairwise[np.eye(len(zs), dtype=bool)] = float('inf')
        spots = []
        for row_n, row in enumerate(pairwise):
            if row_n and row[:row_n].min() < d_min:
                pairwise[:, row_n] = float('inf')
            else:
                spots.append(row_n)
        return spots

    def dedupe_(self, d_min):
        mask_spots = self.dedupe(self._zs[self.z_mask], d_min=d_min)
        spots = self.indices[mask_spots]
        self.z_mask[:] = False
        self.z_mask[spots] = True

    def sparse_extend(self, images, d_min, **meta):
        zs = self.embed_fn(images)
        spots = self.dedupe(images=zs, d_min=d_min)
        zs = zs[spots]
        ds = self.to_goal(zs_2=zs)
        images = images[spots]
        meta = {k: v[spots] for k, v in meta.items()}
        if ds.size == 0:
            return self.extend(zs, images=images, **meta)
        else:
            m = ds.min(axis=0) >= d_min
            if m.any():
                return self.extend(zs[m], images=images[m], **{k: v[m] for k, v in meta.items()})

    def update_zs(self):
        self._zs[self.z_mask] = self.embed_fn(np.stack(self['images']))

    def update_edges(self):
        pairwise = self.pairwise
        l = len(pairwise)
        mask = pairwise >= self.d_max
        pairwise[mask] = float('inf')
        pairwise[np.eye(l, dtype=bool)] = float('inf')
        if l <= self.k:
            knn = np.arange(l, dtype=int).repeat(l).reshape(l, l).T
        else:
            knn = np.argpartition(pairwise, self.k, axis=-1)[:, :self.k]
        indices = self.indices
        for ind, ds, nn, m in zip(indices, pairwise, knn, mask):
            self.ds[ind, :len(nn)] = ds[nn]
            self.adj[ind, :len(nn)] = indices[nn]

    @property
    def edges(self):
        return EdgeView(indices=self.indices, adj=self.adj, z_mask=self.z_mask, ds=self.ds, d_max=self.d_max)

    def neighbors(self, n):
        return [n for n, d in zip(self.adj[n], self.ds[n]) if d < self.d_max]
