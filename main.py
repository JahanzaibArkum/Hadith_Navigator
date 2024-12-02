import os
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
from langchain.chains import RetrievalQA
import streamlit as st

# Set page configuration for Streamlit
st.set_page_config(page_title="Hadith Chatbot", page_icon="🌟", layout="centered")

# Set HuggingFace Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# File paths
CHROMA_DB_PATH = "/content/chroma_db"
USER_DB_PATH = "/content/Book1.csv"
CSV_PATH = '/content/sahih_bukhari_hadiths (1).csv'

# Ensure user database exists
if not os.path.exists(USER_DB_PATH):
    pd.DataFrame(columns=["no", "user", "password"]).to_csv(USER_DB_PATH, index=False)

# Cache the vectorstore initialization
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_DB_PATH):
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

    df = pd.read_csv(CSV_PATH)
    documents = [
        Document(page_content=row['Hadith English'], metadata={
            "reference": row['Reference'],
            "book_number": row['Book Number'],
            "page_number": row['Page Number'],
            "hadith_arabic": row['Hadith Arabic']
        }) for _, row in df.iterrows()
    ]
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DB_PATH)
    vectorstore.persist()
    return vectorstore

vectorstore = load_vectorstore()

# Cache the model initialization
@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base"
    return HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0, "max_new_tokens": 200})

llm = load_model()

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def ask_chatbot(question):
    relevant_docs = vectorstore.similarity_search(question, k=3)
    if not relevant_docs:
        return "No relevant Hadiths found.", []

    context = "\n".join([
        f"Hadith: {doc.page_content}\nReference: {doc.metadata['reference']}"
        for doc in relevant_docs
    ])
    response = retrieval_qa.invoke({
        "query": f"Based on the following Hadiths, answer the question:\n\n{context}\n\nQ: {question}\nA:"
    })
    answer = response.get('result', 'Unable to generate a response.')
    return answer, relevant_docs

def authenticate(user, password):
    df = pd.read_csv(USER_DB_PATH)

    # Convert columns to string to prevent errors with .str accessor
    df["user"] = df["user"].fillna("").astype(str).str.strip()
    df["password"] = df["password"].fillna("").astype(str).str.strip()

    # Strip input to ensure no spaces are included
    user = user.strip()
    password = password.strip()

    # Check if the credentials exist in the dataframe
    return ((df["user"] == user) & (df["password"] == password)).any()

def register_user(user, password):
    df = pd.read_csv(USER_DB_PATH)
    # Normalize by stripping spaces
    df["user"] = df["user"].fillna("").str.strip()

    if (df["user"] == user).any():
        return False  # User already exists
    new_entry = pd.DataFrame({"no": [len(df) + 1], "user": [user], "password": [password]})
    new_entry.to_csv(USER_DB_PATH, mode='a', header=False, index=False)
    return True

# Streamlit UI
st.title("🌟 Hadith Chatbot Login 🌟")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.header("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.header("📝 Sign Up")
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            if register_user(new_username, new_password):
                st.success("Account created successfully! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different username.")

else:
    st.header("📖 Welcome to the Hadith Chatbot!")
    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if question:
            answer, relevant_documents = ask_chatbot(question)
            st.write("Chatbot Answer:", answer)
            for doc in relevant_documents:
                st.write(f"Hadith (English): {doc.page_content}")
                st.write(f"Reference: {doc.metadata['reference']}")
                st.write("-" * 50)
        else:
            st.error("Please enter a question.")

