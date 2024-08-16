import pandas as pd
from pathlib import Path
from langchain_openai import OpenAI
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define paths
input_path = "C:\\Rohan Workplace\\Rohan's Second Brain\\Dr. Reddys"
excel_path = "C:\\Rohan Workplace\\Local Retrieval Augmented Generation\\test.xlsx"
fulldir = Path(input_path).resolve()

# Load and prepare documents
dirloader = DirectoryLoader(fulldir, glob='**/*.pdf', loader_cls=PyPDFLoader)
dirdata = dirloader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
splits = text_splitter.split_documents(dirdata)
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs))
retriever = vectorstore.as_retriever()

# Custom prompt template
template = """Be a friendly helper to your user, have a good conversation with your user and help him/her/them with their queries of the documents. Use the provided pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Answer based on what the user has asked and do not hallucinate.

CONTEXT: ```{context}```
QUESTION: {question}

HELPFUL ANSWER:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(splits):
    return "\n\n".join(split.page_content for split in splits)

def process_questions_from_excel(file_path: str):
    df = pd.read_excel(file_path)
    if 'Questions' not in df.columns:
        raise ValueError("The Excel file must contain a column named 'Questions'.")
    responses = []
    for question in df['Questions']:
        print(f"\nProcessing question: {question}")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | client
            | StrOutputParser()
        )
        response = rag_chain.run(question)
        responses.append(response)
        print(f"Response: {response}")
    return responses

# Run the pipeline
responses = process_questions_from_excel(excel_path)

