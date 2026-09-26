import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

from src.document_loader import get_documents


# =========================================================
# CONFIGURACIÓN
# =========================================================

K = 3

DB_PATH = r"C:\Ai Proyecto\RAG\rag-evaluation-lab\db"

EVAL_PATH = Path(__file__).parent / "eval_dataset.json"


# =========================================================
# CARGAR GOLDEN DATASET
# =========================================================

with open(EVAL_PATH, encoding="utf-8") as f:
    eval_dataset = json.load(f)


# =========================================================
# CREAR RETRIEVERS
# =========================================================

# ----- Dense Retriever -----

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


def retrieve_dense(query):
    return vector_store.similarity_search(
        query=query,
        k=K
    )


# ----- BM25 Retriever -----

documents = get_documents()

bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    k=K
)


def retrieve_bm25(query):
    return bm25_retriever.invoke(query)


# =========================================================
# EVALUACIÓN
# =========================================================

def evaluate_retriever(name, retrieve_function):

    recalls = []
    precisions = []
    reciprocal_ranks = []

    print(f"\n=== {name} ===\n")

    for item in eval_dataset:

        query = item["query"]
        relevant_docs = item["relevant_docs"]

        # Cada retriever busca de forma diferente,
        # pero ambos devuelven List[Document]
        results = retrieve_function(query)

        retrieved_docs = [
            Path(doc.metadata["source"]).name
            for doc in results
        ]

        relevant_retrieved = (
            set(relevant_docs)
            & set(retrieved_docs)
        )

        # Recall@K
        recall = (
            len(relevant_retrieved)
            / len(relevant_docs)
        )

        # Precision@K
        precision = (
            len(relevant_retrieved)
            / len(retrieved_docs)
        )

        # Reciprocal Rank
        rr = 0

        for rank, doc in enumerate(
            retrieved_docs,
            start=1
        ):
            if doc in relevant_docs:
                rr = 1 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(rr)

        print(
            f"{item['id']} | "
            f"Recall@{K}: {recall:.2f} | "
            f"Precision@{K}: {precision:.2f} | "
            f"RR: {rr:.2f}"
        )

        print(f"Esperados:   {relevant_docs}")
        print(f"Recuperados: {retrieved_docs}")
        print()

    # -------------------------
    # Métricas globales
    # -------------------------

    mean_recall = sum(recalls) / len(recalls)
    mean_precision = sum(precisions) / len(precisions)
    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print(f"=== RESULTADOS {name} ===")
    print(f"Recall@{K}:    {mean_recall:.3f}")
    print(f"Precision@{K}: {mean_precision:.3f}")
    print(f"MRR:           {mrr:.3f}")

    return {
        "name": name,
        "recall": mean_recall,
        "precision": mean_precision,
        "mrr": mrr
    }


# =========================================================
# EJECUTAR EXPERIMENTOS
# =========================================================

dense_metrics = evaluate_retriever(
    name="DENSE",
    retrieve_function=retrieve_dense
)

bm25_metrics = evaluate_retriever(
    name="BM25",
    retrieve_function=retrieve_bm25
)


# =========================================================
# COMPARACIÓN
# =========================================================

print("\n=== COMPARACIÓN ===")

print(
    f"{'Retriever':<12}"
    f"{'Recall':<12}"
    f"{'Precision':<12}"
    f"{'MRR':<12}"
)

print("-" * 48)

for metrics in [dense_metrics, bm25_metrics]:

    print(
        f"{metrics['name']:<12}"
        f"{metrics['recall']:<12.3f}"
        f"{metrics['precision']:<12.3f}"
        f"{metrics['mrr']:<12.3f}"
    )