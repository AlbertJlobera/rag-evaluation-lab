import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.rag import ask_rag


load_dotenv()


# =========================================================
# CONFIGURACIÓN
# =========================================================

EVAL_PATH = Path(__file__).parent / "eval_dataset.json"

# Cambia entre:
# "dense"
# "reranker"
RETRIEVAL_MODE = "dense"


def get_eval_dataset():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_doc_id(doc):
    source = doc.metadata["source"]
    return source.replace("\\", "/").split("/")[-1]


# =========================================================
# SALIDA ESTRUCTURADA DEL JUEZ
# =========================================================

class GenerationEvaluation(BaseModel):

    correctness: int = Field(
        ge=0,
        le=2,
        description=(
            "0 = respuesta incorrecta; "
            "1 = parcialmente correcta o incompleta; "
            "2 = completamente correcta y suficientemente completa."
        )
    )

    faithfulness: int = Field(
        ge=0,
        le=2,
        description=(
            "0 = contiene afirmaciones importantes no respaldadas "
            "por el contexto; "
            "1 = mayormente respaldada pero contiene algún problema menor; "
            "2 = todas las afirmaciones están respaldadas por el contexto."
        )
    )

    relevance: int = Field(
        ge=0,
        le=2,
        description=(
            "0 = no responde a la pregunta; "
            "1 = responde parcialmente o contiene información "
            "claramente irrelevante; "
            "2 = responde directamente a la pregunta."
        )
    )

    reasoning: str = Field(
        description="Explicación breve de las puntuaciones asignadas."
    )


# =========================================================
# LLM JUEZ
# =========================================================

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_judge = judge_llm.with_structured_output(
    GenerationEvaluation
)


# =========================================================
# PROMPT DEL JUEZ
# =========================================================

judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un evaluador de sistemas RAG.

Debes evaluar una respuesta generada utilizando tres criterios
independientes: correctness, faithfulness y relevance.

Utiliza siempre esta escala:

0 = incorrecto o incumple claramente el criterio.
1 = parcialmente correcto, incompleto o presenta problemas menores.
2 = correcto y suficientemente completo.


CORRECTNESS

Compara la respuesta generada con la respuesta esperada.

Evalúa si la respuesta generada contiene correctamente
la información esencial necesaria para responder a la pregunta.

No es necesario que utilice exactamente las mismas palabras
que la respuesta esperada.

No penalices una respuesta simplemente por ser más breve
si contiene toda la información necesaria.

Una respuesta correcta pero que omite información importante
debe recibir 1.


FAITHFULNESS

Comprueba si las afirmaciones realizadas en la respuesta generada
están respaldadas por el contexto recuperado.

2 = todas las afirmaciones están respaldadas.
1 = existe algún problema menor o afirmación parcialmente respaldada.
0 = contiene afirmaciones importantes que no están respaldadas
por el contexto.

No utilices la respuesta esperada para evaluar faithfulness.
Utiliza únicamente el contexto recuperado.


RELEVANCE

Evalúa si la respuesta generada responde directamente
a la pregunta realizada.

2 = responde directamente.
1 = responde parcialmente o incluye información claramente innecesaria.
0 = no responde realmente a la pregunta.

La información adicional pertinente no debe penalizarse.


Evalúa los tres criterios de forma independiente
y explica brevemente el motivo de las puntuaciones.
"""
    ),
    (
        "human",
        """
PREGUNTA:
{question}

RESPUESTA ESPERADA:
{expected_answer}

CONTEXTO RECUPERADO:
{context}

RESPUESTA GENERADA:
{generated_answer}
"""
    )
])


# =========================================================
# RETRIEVAL DENSE
# =========================================================

def ask_rag_dense(question):
    """
    Utilizamos los mismos componentes del RAG,
    pero saltándonos el reranker.

    Dense retrieval -> Top 3 -> LLM
    """

    # Importamos los componentes que ya tenemos en src.rag
    # para no duplicar configuración.
    from src.rag import vector_store, prompt, llm

    documents = vector_store.similarity_search(
        query=question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "documents": documents
    }


# =========================================================
# SELECCIÓN DEL PIPELINE
# =========================================================

def generate_answer(question):

    if RETRIEVAL_MODE == "dense":
        return ask_rag_dense(question)

    if RETRIEVAL_MODE == "reranker":
        return ask_rag(question)

    raise ValueError(
        f"RETRIEVAL_MODE no válido: {RETRIEVAL_MODE}"
    )


# =========================================================
# EVALUACIÓN
# =========================================================

def evaluate_generation():

    dataset = get_eval_dataset()

    results = []

    for item in dataset:

        question = item["query"]
        expected_answer = item["expected_answer"]

        # -------------------------------------------------
        # 1. Nuestro RAG genera la respuesta
        # -------------------------------------------------

        rag_result = generate_answer(question)

        generated_answer = rag_result["answer"]
        documents = rag_result["documents"]

        # Reconstruimos exactamente el contexto
        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        retrieved_docs = [
            get_doc_id(doc)
            for doc in documents
        ]

        # -------------------------------------------------
        # 2. Construimos el prompt del juez
        # -------------------------------------------------

        judge_messages = judge_prompt.invoke({
            "question": question,
            "expected_answer": expected_answer,
            "context": context,
            "generated_answer": generated_answer
        })

        # -------------------------------------------------
        # 3. LLM-as-a-Judge
        # -------------------------------------------------

        evaluation = structured_judge.invoke(
            judge_messages
        )

        results.append(evaluation)

        # -------------------------------------------------
        # 4. Resultado individual
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print(
            f"{item['id']} | "
            f"{question} | "
            f"{RETRIEVAL_MODE.upper()}"
        )
        print("-" * 70)

        print(f"Documentos: {retrieved_docs}")

        print("\nRespuesta esperada:")
        print(expected_answer)

        print("\nRespuesta generada:")
        print(generated_answer)

        print("\nEvaluación:")
        print(
            f"Correctness:  "
            f"{evaluation.correctness}/2"
        )
        print(
            f"Faithfulness: "
            f"{evaluation.faithfulness}/2"
        )
        print(
            f"Relevance:    "
            f"{evaluation.relevance}/2"
        )

        print("\nReasoning:")
        print(evaluation.reasoning)


    # =====================================================
    # MÉTRICAS AGREGADAS
    # =====================================================

    total = len(results)

    correctness_score = sum(
        result.correctness
        for result in results
    ) / total

    faithfulness_score = sum(
        result.faithfulness
        for result in results
    ) / total

    relevance_score = sum(
        result.relevance
        for result in results
    ) / total


    # Escala 0-2 -> 0-1

    correctness_normalized = correctness_score / 2
    faithfulness_normalized = faithfulness_score / 2
    relevance_normalized = relevance_score / 2


    # =====================================================
    # RESULTADOS FINALES
    # =====================================================

    print("\n" + "=" * 70)
    print(
        f"=== GENERATION EVALUATION: "
        f"{RETRIEVAL_MODE.upper()} ==="
    )
    print("=" * 70)

    print("\nMedia (0-2):")

    print(
        f"Correctness:  "
        f"{correctness_score:.3f} / 2"
    )

    print(
        f"Faithfulness: "
        f"{faithfulness_score:.3f} / 2"
    )

    print(
        f"Relevance:    "
        f"{relevance_score:.3f} / 2"
    )

    print("\nNormalizado (0-1):")

    print(
        f"Correctness:  "
        f"{correctness_normalized:.3f}"
    )

    print(
        f"Faithfulness: "
        f"{faithfulness_normalized:.3f}"
    )

    print(
        f"Relevance:    "
        f"{relevance_normalized:.3f}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    evaluate_generation()