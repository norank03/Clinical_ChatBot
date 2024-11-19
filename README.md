I created a clinical chatbot to answer questions related to disease symptoms. I used the Cohere API to generate embeddings, where each row of disease and symptom data was embedded as a single vector. When retrieving information based on a user’s question, the dataset was too large to fit within the context length of the LLaMA model. To handle this, I used the Pegasus summarizer to condense the retrieved information to its main points. I then provided this summarized content as context for the LLaMA model, enabling it to answer the user's question effectively.


![Screenshot (1259)](https://github.com/user-attachments/assets/45ace3a9-5459-4957-93e6-d416c8d23a0d)


![Screenshot (1261)](https://github.com/user-attachments/assets/68590d5b-7a08-4129-8501-30d03f05fbb5)
