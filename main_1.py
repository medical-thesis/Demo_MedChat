import os
from dotenv import load_dotenv

from data_loader import DataPreparation, VectorStore
from api_manager import APIKeyManager
from query_process.cohere import Prompt
from history import ChatHistory
from calc_similarity import Rerank, ZeroShot
from langchain_community.llms import Cohere

from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

def generate_response(query: str, query_type: str,vector_db: FAISS, llm: Cohere, 
                    rerank: Rerank, zero_shot: ZeroShot, focus_areas: list) -> str:
    """
    Generate a response to the query using the provided LLM and vector database.
    """
    if query_type == "disease_info":
        focus_area = zero_shot.route_query_zero_shot(query=query, store_focus_areas=focus_areas)
        print("Focus Area: ", focus_area)
        
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {
                    "focus_area": focus_area
                }
            }
        ) 
    elif query_type == "diagnosis_query":
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 10,
            }
        ) 
    else: 
        prompt = Prompt.general_query_prompt(query=query)
        response = llm.invoke(prompt)
        return response
    
    retriever_docs = retriever.invoke(query)
    
    if not retriever_docs:
        return "Sory, I couldn't find the relevant information. Please provide more details."
    
    reranked_docs = rerank.rerank(query, retriever_docs, top_k=3)   
            
    if not reranked_docs:
        return "I found some related documents, but I'm not confident in the answer. Are you able to provide more details?"
    
    context = "\n".join([f"- {doc.page_content}" for doc in reranked_docs])
    
    prompt = f"""
        You are a medical assistant providing concise and responsible health advice.
        The following context has been retrieved and reranked based on relevance. Use only this context to answer the question.
        If the context does not provide enough information, respond with: "I couldn't find the relevant information. Please provide more details."
        
        Context: {context}
        Question: {query}
        Answer:
        """

    response = llm.invoke(prompt)
    return response
    

if __name__ == "__main__":
    
    # data path
    data_path = "data/medquad.csv"
    database_path = "database/vectorstore"

    # Initialize data preparation and vector store
    if os.path.exists(database_path):
        data_preparation = DataPreparation(db_path= data_path)
        focus_areas = data_preparation.get_focus_area()
        
        # Load the vector database
        vector_store = VectorStore(embedding_model=os.getenv("EMBEDDING_MODEL"),
                                embedding_api_key=os.getenv("EMBEDDING_API_KEY"))
        vector_db = vector_store.load_vectordb(load_path= database_path)
    else:
        data_preparation = DataPreparation(db_path= data_path)
        docs, focus_areas = data_preparation.prepare_data()
        
        vector_store = VectorStore(embedding_model=os.getenv("EMBEDDING_MODEL"),
                                embedding_api_key=os.getenv("EMBEDDING_API_KEY"))
        vector_db = vector_store.create_vectordb(docs, database_path)
        
    # Initialize chat history
    chat_history = ChatHistory()
        
    # Initialize llm
    llm = Cohere(model="command",
                    cohere_api_key= os.getenv("COHERE_API_KEY"),
                    temperature=0.5,
    )
        
    rerank = Rerank(model_name=os.getenv("EMBEDDING_MODEL"))
    
    zero_shot = ZeroShot(model_name=os.getenv("EMBEDDING_MODEL"))
    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------") 
    # Initialize the chatbot
    print("Welcome to the Medical Assistant Chatbot!")
    print("Chatbot is ready. Type 'exit' to quit.")
    # Main loop for user input

    while True:
        query = str(input())
        if query.lower() == "exit":
            break
        print("Query: ", query)
        
        # Normalize the query
        context = chat_history.get_context_chathistory()
        prompt_normalization = Prompt.query_normalization_prompt(query=query, context=context)
        query_normalized = llm.invoke(prompt_normalization)
        print("Query Normalized: ", query_normalized)
        
        # Classify the query type
        prompt_classify = Prompt.query_classifier_prompt(query=query_normalized)
        query_type = llm.invoke(prompt_classify)
        print("Query Type: ", query_type)
        
        # Generate a response based on the query type
        response = generate_response(query=query_normalized, 
                                     query_type=query_type, 
                                     vector_db=vector_db, llm=llm, 
                                     rerank=rerank, 
                                     zero_shot=zero_shot, 
                                     focus_areas=focus_areas)
       

        print("Response: ", response)
        
        # Update chat history
        chat_history.add_message({"query": query, "response": response})
        
        print("----------------------------------------------------------------------------------------------------")