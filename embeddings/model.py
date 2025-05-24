from gensim.models.fasttext import FastText, load_facebook_model
from gensim.utils import simple_preprocess
import numpy as np
import compress_fasttext

class Embedder:
    def __init__(self, model_path:str, dummy:bool = False):
        if dummy:
            class DummyModel:
                vector_size = 300
                wv = {}
                def __contains__(self, key): return False
                def __getitem__(self, key): return np.zeros(self.vector_size)

            self.model = DummyModel()
        
        else:
            self.model: FastText = self.load_fasttext_model(model_path)
        self.processor = self.preprocess

    def preprocess(self, text: str):
        if text is None:
            return []
        return simple_preprocess(text)

    def load_fasttext_model(self,path:str):
        # return load_facebook_model(path)
        return compress_fasttext.models.CompressedFastTextKeyedVectors.load(path)


    def get_vector_from_text(self, text: str) -> np.ndarray:
        """
        Return mean embedding for the text.
        """
        tokens = simple_preprocess(text)
        vectors = [self.model[word] for word in tokens]
        if vectors:
            return np.mean(vectors, axis=0)

        dim = getattr(self.model, "vector_size", 300)
        return np.zeros(dim)