 

# Overview

- [**planning.py**](./planning.md) includes examples on how to use this graph with graph_search.
- [**graph_sparsification.py**](graph_sparsification.md) examples on how to sparsify a graph
- [**batch_sparsification.py**](batch_sparsification.md)  efficiently sparsify a graph with as a batch
- [**incremental_api.py**](incremental_api.md) examples on how to extend with sparsity

The sparse graph implementation in this repository makes the assumption that the graph is non-directional, therefore the weighted edges would embed the graph on a Riemann manifold, where the geodesics coincide with shortest-path-distances on the graph. We force a projection from this manifold to an $\ell^p$ sub manifold, by forcing the value function to take the form of $V(g, g') = \Vert \varphi(g) - \varphi(g') \Vert_p$, where $\varphi$ is an embedding.

If the graph are directional, or the value function is not factoried into an embedding function and an $\ell^p$ metric, then this implementation would have to compute the pairwise distance matrix without using the cached latent vectors. This is not necessarily that slow. 