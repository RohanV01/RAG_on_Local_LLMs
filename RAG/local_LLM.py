from pathlib import Path
from langchain_openai import OpenAI
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

input_path = r"C:\\Rohan Workplace\\Local Retrieval Augmented Generation\\test_md"
fulldir = Path(input_path)

# Ensure the path is absolute
if not fulldir.is_absolute():
    fulldir = fulldir.resolve()

# Instantiate the DirectoryLoader to handle PDF files specifically
dirloader = DirectoryLoader(fulldir, glob='**/*.md', loader_cls=UnstructuredMarkdownLoader)

# Load the documents from the specified directory
dirdata = dirloader.load()

# Print a message to confirm the PDF documents have been loaded successfully
print("MD documents loaded successfully.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
print("Documents chunked")
splits = text_splitter.split_documents(dirdata)
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}


vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings(model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs))

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = client
template = """Be a friendly helper to your user, have a good conversation with your user and help him/her/them with their queries of the docuemnts.Use the provided pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Answer based on what the user has asked and do not hallucinate

CONTEXT: ```{context}```
QUESTION: {question}

HELPFUL ANSWER:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(splits):
    return "\n\n".join(split.page_content for split in splits)


def enter_question():
    print("about to invoke the rag_chain")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    question = input("Enter your prompt: ")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("just finished invoking the rag_chain")

while True:
    enter_question()

