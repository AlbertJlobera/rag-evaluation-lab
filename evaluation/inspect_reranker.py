from pathlib import Path

from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from evaluation.evaluate import get_json


# --------------------------------------------------
# Configuración
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db"

CANDIDATE_K = 5

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)


# --------------------------------------------------
# Helper para obtener el nombre del documento
# --------------------------------------------------

def get_doc_id(doc):
    source = doc.metadata["source"]
    return source.replace("\\", "/").split("/")[-1]


# --------------------------------------------------
# Dataset de evaluación
# --------------------------------------------------

eval_ds = get_json()


# --------------------------------------------------
# Evaluamos visualmente cada query
# --------------------------------------------------

for item in eval_ds:

    query = item["query"]
    relevant_docs = item["relevant_docs"]

    # 1. Dense Retrieval -> Top 5 candidatos
    dense_results = vector_store.similarity_search(
        query=query,
        k=CANDIDATE_K
    )

    # 2. Creamos parejas [query, documento]
    #    que recibirá el Cross-Encoder
    pairs = [
        [query, doc.page_content]
        for doc in dense_results
    ]

    # 3. Cross-Encoder calcula relevancia
    scores = reranker.predict(pairs)

    # 4. Asociamos documento + score
    #    y ordenamos de mayor a menor
    reranked_results = sorted(
        zip(dense_results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------
    # Resultados
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print(f"RELEVANTES: {relevant_docs}")

    print("\n--- DENSE TOP 5 ---")

    for rank, doc in enumerate(dense_results, start=1):

        doc_id = get_doc_id(doc)

        relevant = (
            "RELEVANTE"
            if doc_id in relevant_docs
            else ""
        )

        print(
            f"{rank}. {doc_id:<30} {relevant}"
        )

    print("\n--- RERANKED TOP 5 ---")

    for rank, (doc, score) in enumerate(
        reranked_results,
        start=1
    ):

        doc_id = get_doc_id(doc)

        relevant = (
            "RELEVANTE"
            if doc_id in relevant_docs
            else ""
        )

        print(
            f"{rank}. {doc_id:<30} "
            f"score={score:.4f} "
            f"{relevant}"
        )