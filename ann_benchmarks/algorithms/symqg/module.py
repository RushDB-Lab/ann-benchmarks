import symphonyqg as symqg
import numpy as np

from ..base.module import BaseANN


class Symqg(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        # self.p = symqg.Index(space=self.metric, dim=len(X[0]))
        N, D = X.shape
        self.p = symqg.Index(
                index_type="QG",
                metric="L2",
                num_elements=N,
                dimension=D,
                degree_bound=32,
            )
        # self.p.init_index(
        #     max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        # )
        self.p.build_index(data=X, EF=200, num_iter=3)
        # data_labels = np.arange(len(X))
        # self.p.add_items(np.asarray(X), data_labels)
        # self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        print(ef)
        self.p.set_ef(ef)
        self.name = "symqg (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        print(v)
        print(n)
        print(self.p.search(np.expand_dims(v, axis=0), k = 10)[0])
        return self.p.search(query=np.expand_dims(v, axis=0), k=10)[0][0]

    def freeIndex(self):
        del self.p
