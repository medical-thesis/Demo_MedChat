class Prompt:
    def general_query_prompt(query: str) -> str:
        """
        Define the prompt template for generating a response.
        """
        prompt = f""" 
        Please classify the question into one of the following categories and provide the exact corresponding response: 

        - If the query is related to a greeting, respond only with: "Hi, I'm a medical assistant. How can I help you today?" 
        - If the query is related to goodbyes, respond only with: "Goodbye! Have a great day!" 
        - If the query is expresses gratitude, respond only with: "You're welcome! I'm here to help." 
        - If the query is about the chatbot, respond only with: "I'm a medical assistant here to provide health advice. I am trained to be a helpful and supportive AI assistant, so feel comfortable asking me any questions." 
        - Otherwise, respond only with: "I couldn't find the relevant information." 

        Ensure that your response strictly follows the given options without any additional text. 

        Question: {query} 
        Answer: 
        """
        
        return prompt
    def query_classifier_prompt(query: str) -> str:
        """
        Define the prompt template for query classification.
        """
        prompt = f"""
        You are a medical query classifier. Classify the type of question.

        Question: {query}
        
        Categories:
        - If it is about a specific disease or condition (like 'What is glaucoma?'), respond with: disease_info
        - If it is asking for diagnosis or possible diseases from symptoms (like 'I have a headache, what could it be?'), respond with: diagnosis_query
        - If it is an incomplete query (like 'yes', 'no', 'maybe'), respond with: diagnosis_query
        - If it is an query about common information (like greetings, aking for the assistant: "What is your name?"), respond with: general
        """
        return prompt
    def query_normalization_prompt(query: str, context: str= None) -> str:
        """
        Define the prompt template for query normalization.
        """
        if context:
            prompt= f"""    
                Normalize the following medical query by making it more concise and clear.
                If the original query information is unclear or related to the conversation history, use the conversation history to normalize the query.
                Rephrase if needed and remove any irrelevant information in the original query.
                Focus on maintaining the core meaning and intent of the query.
                Avoid adding any new information or context that is not present in the original query.
                Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                
                
                Example 1:
                    Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                    Normalized Query: "Are dizziness and nausea related to anemia?"
                    
                Example 2:
                    Original query: "Can medicine A be used to treat sore throat? I see different information online."
                    Normalized Query: "Can medicine A treat sore throat?"
                    
                Example 3:
                    Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                    Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                    
                Example 4:
                    Original query: "What is Glaucoma?"
                    Normalized Query: "What is Glaucoma?"
                
                Example 5:
                    Original query: "What causes Glaucoma ?"
                    Normalized Query: "What causes Glaucoma ?"
                    
                Example 6:
                    Original query: "What diseases can headaches be a sign of?"
                    Normalized Query: "What diseases can headaches be a sign of?"
                    
                Example 7:
                    Original query: "I have frequent headaches, is it due to anemia?"
                    Normalized Query: "Is frequent headache related to anemia?"  
                    
                Example 8:
                    Conversation history:
                        - Query: What are the side effects of drug A? 
                        - Response: Drug A can cause drowsiness and dry mouth.
                        - Query: I am allergic to antihistamines, can I use drug A? 
                        - Response: Drug A belongs to the antihistamine group, so you should consult your doctor before using it.

                    Original Query: "I have allergic rhinitis, can I use medicine A?"
                    Normalized Query: "Can I use medicine A for allergic rhinitis?"
                
                Example 9:
                    Conversation history:
                        - Query: What diseases can headaches be a sign of? 
                        - Response: Headaches can be caused by many things such as stress, lack of sleep or serious illnesses such as high blood pressure.
                        - Query: I have frequent headaches, is it due to anemia? 
                        - Response: Anemia can cause headaches, but tests are needed to determine exactly.

                    Original Query: "So if a headache is accompanied by dizziness, what diseases can it be?"
                    Normalized Query: "What diseases can a headache with dizziness be?"

                Given the conversation history:
                {context}
                    
                Original query: {query}
                Normalized query:
                """
        else:
            prompt = f"""
                Normalize the following medical query by making it more concise and clear.
                Rephrase if needed and remove any irrelevant information in the original query.
                Focus on maintaining the core meaning and intent of the query.
                Avoid adding any new information or context that is not present in the original query.
                Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                
                Example 1:
                    Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                    Normalized Query: "Are dizziness and nausea related to anemia?"
                    
                Example 2:
                    Original query: "Can medicine A be used to treat sore throat? I see different information online."
                    Normalized Query: "Can medicine A treat sore throat?"    
                
                Example 3:
                    Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                    Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                
                Example 4:
                    Original query: "What is Glaucoma?"
                    Normalized Query: "What is Glaucoma?"
                
                Example 5:
                    Original query: "What causes Glaucoma ?"
                    Normalized Query: "What causes Glaucoma ?"
                    
                Example 6:
                    Original query: "What diseases can headaches be a sign of?"
                    Normalized Query: "What diseases can headaches be a sign of?"
                    
                Example 7:
                    Original query: "I have frequent headaches, is it due to anemia?"
                    Normalized Query: "Is frequent headache related to anemia?"
           
                
                Original query: {query}
                Normalized query:
                """
                
        return prompt

    def prompt(context: str, query: str) -> str:
        """
        Define the prompt template for generating a response.
        """
        prompt = f"""
            You are a medical assistant providing concise and responsible health advice.
            The following context has been retrieved and reranked based on relevance. Use only this context to answer the question.
            If the context does not provide enough information, respond with: "I couldn't find the relevant information. Please provide more details."
            
            Context: {context}
            Question: {query}
            Answer:
            """
        
        return prompt