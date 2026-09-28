import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

from src.document_loader import get_documents
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# CONFIGURACIÓN
# =========================================================
def get_doc_id(doc):
    source = doc.metadata["source"]

    # Compatible con rutas Windows y Unix/Mac
    return source.replace("\\", "/").split("/")[-1]


# Número de documentos que evaluamos finalmente
FINAL_K = 3

# Número de candidatos que Dense y BM25 entregan a RRF
CANDIDATE_K = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "db"

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
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)


def retrieve_dense(query):
    """
    Dense baseline.
    Devuelve Top 3 para compararlo justamente con los demás.
    """
    return vector_store.similarity_search(
        query=query,
        k=FINAL_K
    )


# ----- BM25 Retriever -----

documents = get_documents()

bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    k=FINAL_K
)


def retrieve_bm25(query):
    """
    BM25 baseline.
    Devuelve Top 3.
    """
    bm25_retriever.k = FINAL_K
    return bm25_retriever.invoke(query)


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    rankings,
    final_k=3,
    c=60
):
    """
    Fusiona varios rankings utilizando RRF.

    rankings:
        Lista con rankings de distintos retrievers.

    final_k:
        Número de documentos finales.

    c:
        Constante de suavizado de RRF.
    """

    scores = {}
    documents_by_id = {}

    for ranking in rankings:

        for rank, doc in enumerate(
            ranking,
            start=1
        ):

            # Usamos el nombre del fichero como ID único
            doc_id = get_doc_id(doc)

            documents_by_id[doc_id] = doc

            if doc_id not in scores:
                scores[doc_id] = 0

            # Score RRF
            scores[doc_id] += 1 / (c + rank)

    # Ordenar documentos por score RRF
    sorted_doc_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # Devolver únicamente Top K final
    return [
        documents_by_id[doc_id]
        for doc_id in sorted_doc_ids[:final_k]
    ]


# =========================================================
# HYBRID RETRIEVER
# =========================================================

def retrieve_hybrid(query):
    """
    Hybrid Retrieval:

    Dense Top 5
         +
    BM25 Top 5
         ↓
        RRF
         ↓
      Top 3
    """

    # Dense candidates
    dense_results = vector_store.similarity_search(
        query=query,
        k=CANDIDATE_K
    )

    # BM25 candidates
    bm25_retriever.k = CANDIDATE_K
    bm25_results = bm25_retriever.invoke(query)

    # Fusionamos ambos rankings
    return reciprocal_rank_fusion(
        rankings=[
            dense_results,
            bm25_results
        ],
        final_k=FINAL_K
    )


# =========================================================
# EVALUACIÓN
# =========================================================

def evaluate_retriever(
    name,
    retrieve_function
):

    recalls = []
    precisions = []
    reciprocal_ranks = []

    print(f"\n=== {name} ===\n")

    for item in eval_dataset:

        query = item["query"]
        relevant_docs = item["relevant_docs"]

        # Todos los retrievers cumplen el mismo contrato:
        #
        # query -> List[Document]
        results = retrieve_function(query)

        retrieved_docs = [
            get_doc_id(doc)
            for doc in results
        ]

        relevant_retrieved = (
            set(relevant_docs)
            & set(retrieved_docs)
        )

        # -------------------------
        # Recall@K
        # -------------------------
        recall = (
            len(relevant_retrieved)
            / len(relevant_docs)
        )
        # -------------------------
        # Precision@K
        # -------------------------

        precision = (
            len(relevant_retrieved)
            / len(retrieved_docs)
        )
        # -------------------------
        # Reciprocal Rank
        # -------------------------

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
            f"Recall@{FINAL_K}: {recall:.2f} | "
            f"Precision@{FINAL_K}: {precision:.2f} | "
            f"RR: {rr:.2f}"
        )

        print(
            f"Esperados:   {relevant_docs}"
        )

        print(
            f"Recuperados: {retrieved_docs}"
        )

        print()

    # =====================================================
    # MÉTRICAS GLOBALES
    # =====================================================

    mean_recall = (
        sum(recalls)
        / len(recalls)
    )

    mean_precision = (
        sum(precisions)
        / len(precisions)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print(
        f"=== RESULTADOS {name} ==="
    )

    print(
        f"Recall@{FINAL_K}:    "
        f"{mean_recall:.3f}"
    )

    print(
        f"Precision@{FINAL_K}: "
        f"{mean_precision:.3f}"
    )

    print(
        f"MRR:           "
        f"{mrr:.3f}"
    )

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

hybrid_metrics = evaluate_retriever(
    name="HYBRID",
    retrieve_function=retrieve_hybrid
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

for metrics in [
    dense_metrics,
    bm25_metrics,
    hybrid_metrics
]:

    print(
        f"{metrics['name']:<12}"
        f"{metrics['recall']:<12.3f}"
        f"{metrics['precision']:<12.3f}"
        f"{metrics['mrr']:<12.3f}"
    )