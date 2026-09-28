from sentence_transformers import CrossEncoder
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "db"
# ----- Dense Retriever -----

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)

# 1. Definimos Q14
query = "¿Qué diferencia hay entre valuation_details y risk_results?"

# 2. Dense obtiene los 5 candidatos
dense_results = vector_store.similarity_search(
    query=query,
    k=5
)

# 3. Construimos las parejas para el Cross-Encoder
pairs = [
    [query, doc.page_content]
    for doc in dense_results
]

scores = reranker.predict(pairs)