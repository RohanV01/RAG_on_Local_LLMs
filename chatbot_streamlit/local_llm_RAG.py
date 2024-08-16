import streamlit as st
from pathlib import Path
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(page_title="3D Bioprinting ChatBot", layout="wide")

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define paths
db_path = "C:\\Rohan Workplace\\Local Retrieval Augmented Generation\\chroma_db_bioinks"

# Load the Chroma DB
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embedding_function = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)
retriever = vectorstore.as_retriever()

# Define Prompt Template
template = """Be a friendly helper to your user, have a good conversation with your user and help him/her/them with their queries of the documents.
Use the provided pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Cite all the sources at the end of each answer.
Keep the answer as concise as possible.
Answer based on what the user has asked and do not hallucinate.

CONTEXT: ```{context}``
QUESTION: {question}

HELPFUL ANSWER:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Utility function to format documents
def format_docs(splits):
    return "\n\n".join(split.page_content for split in splits)

# Function to query RAG pipeline
def query_rag(question, token_display):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | client
        | StrOutputParser()
    )
    response = ""
    try:
        for chunk in rag_chain.stream(question):
            response += chunk
            token_display.markdown(response)
    except Exception as e:
        st.write(f"Error querying RAG pipeline: {e}")
    return response

# Main function for Streamlit app
def main():
    st.title("3D Bioprinting ChatBot")
    st.write("Ask questions about 3D Bioprinting")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        st.write("Configure your chatbot settings here.")

    # Layout using columns
    chat_container, input_container = st.columns([3, 1])

    with input_container:
        question = st.text_input("Enter your prompt:", key="input")
        if st.button("Submit", key="submit"):
            if question:
                with st.spinner('Processing...'):
                    token_display = st.empty()
                    response = query_rag(question, token_display)
                    st.session_state.chat_history.append({"question": question, "response": response})
                    st.experimental_rerun()

    with chat_container:
        st.header("Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**User:** {chat['question']}")
            st.write(f"**Assistant:** {chat['response']}")

if __name__ == "__main__":
    main()
