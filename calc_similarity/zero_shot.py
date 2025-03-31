from sentence_transformers import util, SentenceTransformer
import torch
import os
from dotenv import load_dotenv
load_dotenv()


class ZeroShot:
    def __init__(self, model_name= str(os.getenv("EMBEDDING_MODEL"))):
        """
        Initialize the ZeroShot class with a pre-trained model.
        Parameters:
        - model_name: The name of the pre-trained model to be used for embeddings.
        """
        self.model = SentenceTransformer(model_name)
    
    def embed(self, query):
        """
        Embed the input query using the pre-trained model.
        Parameters:
        - query: The input query to be embedded.
        Returns:
        - The embedded query as a tensor.
        """
        return self.model.encode(query, convert_to_tensor=True).cpu()
    
    def cos_sim(self, query_embedding, doc_embedding):
        """
        Calculate the cosine similarity between the query and document embeddings.
        Parameters:
        - query_embedding: The embedded query.
        - doc_embedding: The embedded document.
        Returns:
        - The cosine similarity score.
        """
        return util.pytorch_cos_sim(query_embedding, doc_embedding).cpu()
    
    def route_query_zero_shot(self, query, store_focus_areas: list):
        """
        Route the query to the most relevant focus area using zero-shot classification.
        Parameters:
        - query: The input query.
        - store_focus_areas: A list of focus areas to classify against.
        Returns:
        - The most relevant focus area based on the query.
        """
        query_embedding = self.embed(query)
        focus_embeddings = self.embed(store_focus_areas)
        cosine_scores = self.cos_sim(query_embedding, focus_embeddings)
        max_score_idx = torch.argmax(cosine_scores).item()
        return store_focus_areas[max_score_idx]
    
    