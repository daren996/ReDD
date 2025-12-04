from sklearn.cluster import KMeans


class Clusterer:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class ClustererGPT(Clusterer):
    def __init__(self, n_clusters):
        super().__init__(n_clusters)
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit_predict(self, embeddings):
        if not isinstance(embeddings, (list, tuple)):
            raise ValueError("Embeddings must be a list or 2D array-like structure.")
        cluster_ids = self.model.fit_predict(embeddings)
        return cluster_ids
