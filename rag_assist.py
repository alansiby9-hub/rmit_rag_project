from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# --- Load and Prepare Data ---
files = ["enrolment.txt", "timetable.txt"]
docs = []
for file in files:
    docs.extend(TextLoader(file).load())

# Split into smaller text chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Load local model (works with Ollama locally, not on Streamlit Cloud)
llm = Ollama(model="phi3:mini")

# --- Define a simple RAG pipeline ---
prompt = PromptTemplate.from_template(
    "You are an RMIT student support assistant. Use the provided context to answer.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def ask_rag(query):
    # Retrieve relevant docs
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build the final prompt
    input_text = prompt.format(context=context, question=query)
    try:
        answer = llm.invoke(input_text)
    except Exception as e:
        answer = "This model cannot be run on Streamlit Cloud. Please run locally with Ollama."
    sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
    return answer, sources
