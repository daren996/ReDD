from openai import OpenAI


def llm_embeddings(string, model="text-embedding-3-small"):
    """
    Generate Text embeddings using ChatGPT models.
    models: 
        text-embedding-3-small    $0.020 / 1M tokens
        text-embedding-3-large    $0.130 / 1M tokens
        ada v2                    $0.100 / 1M tokens
    """
    client_gpt = OpenAI()
    embeddings = client_gpt.embeddings.create(
        input=string,
        model=model
    )  # TODO: make it support DeekSeek
    return embeddings.data[0].embedding


class Vectorizer:
    def __init__(self):
        pass

    def fit_transform(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class DocVectorizer(Vectorizer):
    def __init__(self, embedder=None):
        super().__init__()
        self.embedder = embedder if embedder else llm_embeddings

    def fit_transform(self, documents):
        return [self.embedder(document) for document in documents]
    