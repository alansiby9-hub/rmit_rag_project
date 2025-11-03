from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load Text Files Directly ---
files = ["enrolment.txt", "timetable.txt"]
docs = []
for file in files:
    docs.extend(TextLoader(file).load())

# --- Split Text into Chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

# --- Embeddings + Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# --- Define the LLM ---
llm = Ollama(model="phi3:mini")

# --- Retrieval-Augmented Generation Chain ---
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_rag(query):
    result = chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    return answer, sources
