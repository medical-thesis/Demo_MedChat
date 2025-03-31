import os
import torch
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import util, SentenceTransformer
load_dotenv()

class Rerank:
    def __init__(self, model_name= str(os.getenv("EMBEDDING_MODEL"))):
        """
        Initialize the Rerank class with a pre-trained model.
        Parameters:
        - model_name: The name of the pre-trained model to be used for embeddings.
        """
        self.model = SentenceTransformer(model_name)
    
    def embed(self, query):
        """
        Embed the input query using the pre-trained model.
        Parameters:
        - 
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

    def rerank(self, query, docs, top_k= 5, threshold=0.5):
        """
        Rerank the documents based on their similarity to the query.
        Parameters:
        - query: The input query.
        - docs: A list of documents to be reranked.
        - top_k: The number of top documents to return.
        - threshold: The similarity threshold for filtering documents.
        Returns:
        - A list of tuples containing the document and its similarity score.
        """
        query_embedding = self.embed(query)
        doc_embeddings = self.embed([doc.page_content for doc in docs])
        
        similarities = self.cos_sim(query_embedding, doc_embeddings)
        
        max_score = torch.max(similarities).item()
        
        if max_score < threshold:
            return None
        
        sorted_indices = np.argsort(similarities.squeeze().numpy())[::-1]
        top_indices = sorted_indices[:top_k]
        
        reranked_docs = [docs[i] for i in top_indices]
        
        return reranked_docs