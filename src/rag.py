from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

CANDIDATE_K = 5



load_dotenv()


# =========================================================
# CONFIGURACIÓN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db"

TOP_K = 3


# =========================================================
# MODELOS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un asistente técnico.

Responde a la pregunta utilizando únicamente la información
proporcionada en el contexto.

Si el contexto no contiene información suficiente para responder,
indícalo claramente.

No inventes información.
"""
    ),
    (
        "human",
        """
Contexto:
{context}

Pregunta:
{question}
"""
    )
])


# =========================================================
# RAG
# =========================================================

def ask_rag(question):

    # 1. Dense retrieval -> candidatos
    candidates = vector_store.similarity_search(
        query=question,
        k=CANDIDATE_K
    )

    # 2. Reranking
    pairs = [
        [question, doc.page_content]
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 3. Nos quedamos con Top 3
    documents = [
        doc
        for doc, score in reranked[:TOP_K]
    ]

    # 4. Construimos contexto
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # 5. Construimos prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # 6. Generación
    response = llm.invoke(messages)

    # 7. Devolvemos respuesta + documentos usados
    return {
        "answer": response.content,
        "documents": documents
    }