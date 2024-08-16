import pandas as pd
from pathlib import Path
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import pyautogui
import time

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define paths
db_path = "C:\\Rohan Workplace\\Local Retrieval Augmented Generation\\chroma_db_bioinks"
excel_path = "C:\\Rohan Workplace\\Local Retrieval Augmented Generation\\ibrahim data.xlsx"

# Load the Chroma DB
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embedding_function = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

print("\nLoading Chroma vector store...")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)

retriever = vectorstore.as_retriever()

# Load the custom prompt template
template = """Be a friendly helper to your user, have a good conversation with your user and help him/her/them with their queries of the documents. Use the provided pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Cite all the sources at the end of each answer.
Keep the answer as concise as possible.
Answer based on what the user has asked and do not hallucinate.

CONTEXT: ```{context}```
QUESTION: {question}

HELPFUL ANSWER:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Function to format documents
def format_docs(splits):
    return "\n\n".join(split.page_content for split in splits)

# Load questions from the Excel file
print("Loading questions from Excel file...")
df = pd.read_excel(excel_path)

# Ensure there is a 'Questions' column in the Excel file
if 'Questions' not in df.columns:
    print("Error: The Excel file must contain a column named 'Questions'.")
    raise ValueError("The Excel file must contain a column named 'Questions'.")

questions = df['Questions'].tolist()
responses = []

# Function to query the RAG pipeline
def query_rag(question):
    print(f"\nQuerying RAG pipeline with question: {question}")
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
            print(f"Received chunk: {chunk}")
    except Exception as e:
        print(f"Error querying RAG pipeline: {e}")
    return response

# Simulate manual input for each question
for i, question in enumerate(questions, start=1):
    print(f"\nProcessing question {i}/{len(questions)}: {question}")
    
    # Simulate typing the question into the terminal
    time.sleep(2)  # Wait for 2 seconds before simulating input
    pyautogui.typewrite(question)
    pyautogui.press('enter')
    
    response = query_rag(question)
    responses.append(response)
    print(f"Response: {response}")

# Add the responses to the DataFrame
df['RAG answers'] = responses

# Save the updated DataFrame back to the Excel file
print("\nSaving responses to Excel file...")
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

print("\nResponses have been saved to the Excel file.")
