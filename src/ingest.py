from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from src.document_loader import get_documents

# Raíz del proyecto
project_root = Path(__file__).resolve().parent.parent

# /data/docs dentro del proyecto
db_path = project_root / "db"

# db_path =r"C:\Ai Proyecto\RAG\rag-evaluation-lab\db"

documents = get_documents()



# Semantic Search

# embed = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )


# vector_store = Chroma.from_documents(
#     documents=documents,
#     embedding=embed,
#     persist_directory=db_path
# )

# Keyword Search

retriever = BM25Retriever.from_documents(
    documents=documents,
    k=3
)
query = "¿Qué diferencia hay entre valuation_details y risk_results?"
results = retriever.invoke(query)

retrieved_docs = [
        Path(result.metadata["source"]).name
        for result in results
    ]


print(retrieved_docs)