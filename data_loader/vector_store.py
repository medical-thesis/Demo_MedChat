from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, embedding_model: str, embedding_api_key: str = None):
        """
        Initialize the VectorStore with an embedding model.

        :param embedding_model: The embedding model to be used.
        :param embedding_api_key: API key for the embedding model, if required.
        """
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        if self.embedding_api_key:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model, 
                                                         api_key=embedding_api_key)
        else:
            # If no API key is provided, use the default HuggingFaceEmbeddings
            # Note: Ensure that the model used is compatible with HuggingFaceEmbeddings
            # and does not require an API key.
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
    
    def create_vectordb(self, documents: list, save_path: str) -> FAISS:
        """
        Create a vector database from the documents.

        :param documents: List of Document objects.
        :param save_path: Path to save the vector database.
        :return: FAISS vector store object.
        """
        vectordb = FAISS.from_documents(documents, self.embedding_model)
        vectordb.save_local(save_path)
        return vectordb

    def load_vectordb(self, load_path: str) -> FAISS:
        """
        Load a vector database from the specified path.
        if the path is invalid or the database cannot be loaded, return None.
        """
        try:
            vectordb = FAISS.load_local(load_path, self.embedding_model, allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return None
    