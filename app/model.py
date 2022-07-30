from sentence_transformers import SentenceTransformer, util


class Model:
    """
    Class Embedding Transformer model
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def cosine_similarity(self, sentence1: str, sentence2: str) -> int:
        """
        Method computing embedding of the 2 sentences and returning cosine similarity between them.
        :param sentence1: first sentence
        :param sentence2: second sentence
        :return: cosine similarity as int
        """
        embedding1 = self.model.encode(sentence1)
        embedding2 = self.model.encode(sentence2)
        return int(util.cos_sim(embedding1, embedding2).item() * 100)
