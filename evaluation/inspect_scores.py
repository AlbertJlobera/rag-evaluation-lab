import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()


# =========================================================
# CONFIGURACIÓN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "db"

EVAL_PATH = Path(__file__).parent / "eval_dataset.json"

INSPECT_K = 5


# =========================================================
# CARGAR GOLDEN DATASET
# =========================================================

with open(EVAL_PATH, encoding="utf-8") as f:
    eval_dataset = json.load(f)


# =========================================================
# VECTOR STORE
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)


# =========================================================
# FUNCIÓN PARA NORMALIZAR EL NOMBRE DEL DOCUMENTO
# =========================================================

def get_doc_id(doc):

    source = doc.metadata["source"]

    # Funciona tanto con rutas Windows como Mac/Linux
    return source.replace("\\", "/").split("/")[-1]


# =========================================================
# INSPECCIONAR SCORES
# =========================================================

for item in eval_dataset:

    query_id = item["id"]
    query = item["query"]
    relevant_docs = item["relevant_docs"]

    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=INSPECT_K
    )

    print("\n" + "=" * 70)

    print(f"{query_id}: {query}")

    print(f"Esperados: {relevant_docs}")

    print("-" * 70)

    for rank, (doc, score) in enumerate(
        results,
        start=1
    ):

        doc_name = get_doc_id(doc)

        if doc_name in relevant_docs:
            status = "RELEVANTE"
        else:
            status = "NO RELEVANTE"

        print(
            f"{rank}. "
            f"{doc_name:<30} "
            f"score={score:.4f} "
            f"{status}"
        )