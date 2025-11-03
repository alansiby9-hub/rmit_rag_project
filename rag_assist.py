# rag_assist.py  — works with LangChain 0.2+
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1) Load documents
loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3) Embed + store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(chunks, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 4) LLM via Ollama (make sure you’ve pulled `mistral`: `ollama pull mistral`)
llm = Ollama(model="gemma:2b")





# 5) Prompt + chain (LC 0.2 style)
prompt = ChatPromptTemplate.from_template(
    "You are a helpful RMIT assistant. Use the context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def ask_rag(query: str):
    """Return (answer_text, source_file_list)."""
    answer = chain.invoke(query)
    # New syntax for LangChain 0.2+
    top_docs = retriever.invoke(query)
    sources = [d.metadata.get("source", "unknown") for d in top_docs]
    return answer, sources

