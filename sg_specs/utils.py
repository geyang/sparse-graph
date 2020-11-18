import numpy as np

id2D = lambda xy: xy[:, :2]
l2 = lambda a, b: np.linalg.norm(a - b, ord=2, axis=-1)
