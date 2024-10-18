import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFaceHub  # Corrected import for HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings  # Corrected import for embeddings
from langchain_community.vectorstores import Chroma  # Corrected import for Chroma
from langchain.schema import Document  # Corrected import for Document
import streamlit as st  # Importing Streamlit

# Set environment variables for Hugging Face tokens
os.environ['HF_TOKEN'] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ['HF_TOKEN']

# Load the CSV containing Hadith data
csv_path = '/content/sahih_bukhari_hadiths.csv'
df = pd.read_csv(csv_path)

# Create LangChain documents with metadata from the CSV file
documents = [
    Document(page_content=row['Hadith English'], metadata={
        "reference": row['Reference'],
        "book_number": row['Book Number'],
        "page_number": row['Page Number'],
        "hadith_arabic": row['Hadith Arabic']
    }) for index, row in df.iterrows()
]

# Load the embedding model for vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory='/content/chroma_db')
vectorstore.persist()

print("Chroma vector store rebuilt and saved.")

# Define the model configuration for the language model 
model_id = "meta-llama/Llama-3.2-1B"

# Initialize the LLM from HuggingFaceHub
llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"device_map": "auto"})

# Create the RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Defines how to use the retrieved documents
    retriever=vectorstore.as_retriever()
)

# Streamlit interface
st.title("Hadith Chatbot")
st.write("Ask a question about the importance of prayer in Islam.")

question = st.text_input("Your Question:")
if st.button("Get Answer"):
    if question:
        answer = ask_chatbot(question)
        st.write("Chatbot Answer:", answer)
    else:
        st.write("Please enter a question.")
