import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataPreparation:
    def __init__(self, db_path: str, chunk_size: int = 2000, overlap: int = 200):
        """
        Initialize the DataPreparation with database path, chunk size, and overlap.

        :param db_path: Path to the database file.
        :param chunk_size: Size of each data chunk.
        :param overlap: Overlap between chunks.
        """
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_data(self):
        """
        Load data from the specified database path.
        """
        dt = pd.read_csv(self.db_path)
        dt = dt.dropna(subset=['answer','question', 'focus_area'])
        return dt
    def split_text(self, text: str) -> list:
        """
        Split the text into smaller chunks.

        :param text: Text to be split.
        :return: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len
        )
        return text_splitter.split_text(text)

    def add_documents_to_chunks(self, dt: pd.DataFrame) -> list:
        """
        Add chunks to documents.

        :param dt: DataFrame containing the data.
        :return: List of Document objects.
        """
        documents = []
        for _, row in dt.iterrows():
            text = row['answer']
            # Create metadata dictionary for each row
            metadata= {
                "focus_area": row['focus_area'], 
                "question": row['question']
            }
            chunks = self.split_text(text)
            for chunk in chunks:
                doc = Document(page_content=chunk, 
                               metadata=metadata)
                documents.append(doc)
        return documents   
    
    def store_focus_area(self, dt: pd.DataFrame) -> list:
        """
        Store focus area from the DataFrame.

        :param dt: DataFrame containing the data.
        :return: List of unique focus areas.
        """
        focus_areas = dt['focus_area'].unique().tolist()
        return focus_areas
    
    def prepare_data(self) -> tuple:
        """
        Prepare the data by loading it, splitting it into chunks, and storing focus areas.

        :return: Tuple of documents and focus areas.
        """
        dt = self.load_data()
        documents = self.add_documents_to_chunks(dt)
        focus_areas = self.store_focus_area(dt)
        return documents, focus_areas
    
    def get_focus_area(self) -> list:
        """
        Get focus area from the DataFrame.

        :param dt: DataFrame containing the data.
        :return: List of unique focus areas.
        """
        dt = self.load_data()
        focus_areas = dt['focus_area'].unique().tolist()
        return focus_areas