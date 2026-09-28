# RAG Evaluation Lab

A practical Retrieval-Augmented Generation (RAG) evaluation project focused on measuring retrieval quality, generation quality, latency, and architectural trade-offs.

Rather than building another simple "chat with your documents" application, this project explores a more important engineering question:

> **How do we know whether a RAG architecture is actually improving the final answer?**

The project compares multiple retrieval strategies — Dense Retrieval, BM25, Hybrid Search with Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking — using a custom golden dataset.

It then evaluates the complete RAG pipeline using an LLM-as-a-Judge approach.

---

## Objectives

The main goals of this project are:

- Build a complete RAG pipeline.
- Compare different retrieval strategies.
- Measure retrieval quality using standard Information Retrieval metrics.
- Evaluate generated answers separately from retrieval.
- Analyze quality vs latency trade-offs.
- Make architectural decisions based on experimental results rather than assumptions.

---

## Architecture

### Final RAG architecture

```text
User Query
    ↓
Dense Retrieval
    ↓
Top-K Documents
    ↓
Context Construction
    ↓
LLM
    ↓
Generated Answer
```

During experimentation, an alternative architecture using reranking was also evaluated:

```text
User Query
    ↓
Dense Retrieval (Top 5)
    ↓
Cross-Encoder Reranker
    ↓
Top 3 Documents
    ↓
Context Construction
    ↓
LLM
    ↓
Generated Answer
```

---

## Dataset

The project uses a small synthetic dataset representing a financial risk platform.

The documents cover topics such as:

- System architecture
- Equity method calculations
- Exposure calculations
- Credit spread calculations
- End-of-day processing
- Incidents
- Reconciliation
- Data sources
- Deployment
- Monitoring
- Security
- Troubleshooting

The dataset was intentionally designed with overlapping information between documents to make retrieval less trivial.

For this experiment, each `.txt` file is treated as one retrievable document.

This was a deliberate baseline decision because the documents are small and topic-specific.

---

## Golden Evaluation Dataset

A custom evaluation dataset contains 15 questions.

Each test case includes:

```json
{
  "id": "Q14",
  "query": "¿Qué diferencia hay entre valuation_details y risk_results?",
  "relevant_docs": ["architecture.txt"],
  "expected_answer": "valuation_details almacena los resultados de detalle y risk_results almacena los resultados agregados."
}
```

The dataset supports two independent evaluation layers:

```text
query
├── relevant_docs
│       ↓
│   Retrieval Evaluation
│
└── expected_answer
        ↓
    Generation Evaluation
```

This separation makes it possible to identify whether a failure originates in retrieval or generation.

---

# Retrieval Experiments

Four retrieval strategies were evaluated.

## 1. Dense Retrieval

Semantic retrieval using:

- OpenAI `text-embedding-3-small`
- Chroma vector database

```text
Query
  ↓
Embedding
  ↓
Vector similarity
  ↓
Top-K documents
```

---

## 2. BM25

A sparse lexical retrieval baseline using BM25.

```text
Query
  ↓
Token matching
  ↓
BM25 ranking
  ↓
Top-K documents
```

This provides a useful comparison between semantic and lexical retrieval.

---

## 3. Hybrid Retrieval

Dense Retrieval and BM25 were combined using Reciprocal Rank Fusion (RRF).

```text
              ┌── Dense Top 5
Query ────────┤
              └── BM25 Top 5
                      ↓
                     RRF
                      ↓
                   Top 3
```

RRF combines rankings rather than directly comparing Dense and BM25 scores.

---

## 4. Cross-Encoder Reranking

Dense Retrieval was used for candidate generation and a Cross-Encoder for second-stage ranking.

```text
Query
  ↓
Dense Retrieval
  ↓
Top 5 Candidates
  ↓
Cross-Encoder
  ↓
Reranking
  ↓
Top 3
```

The final experiment used:

`BAAI/bge-reranker-v2-m3`

A multilingual reranker was selected because the evaluation queries are written in Spanish.

---

# Retrieval Evaluation

The following metrics were used:

### Recall@K

Measures how many relevant documents were successfully retrieved.

```text
relevant documents retrieved
────────────────────────────
 total relevant documents
```

### Precision@K

Measures how many retrieved documents were actually relevant.

```text
relevant documents retrieved
────────────────────────────
   retrieved documents
```

### Mean Reciprocal Rank (MRR)

Measures how early the first relevant document appears in the ranking.

Higher MRR means relevant information tends to appear closer to the top.

---

## Retrieval Results

| Retriever | Recall@3 | Precision@3 | MRR |
|---|---:|---:|---:|
| Dense | 0.917 | 0.467 | 0.789 |
| BM25 | 0.472 | 0.244 | 0.489 |
| Hybrid (Dense + BM25 + RRF) | 0.811 | 0.400 | 0.756 |
| Dense + Cross-Encoder Reranker | **0.950** | **0.467** | **0.900** |

The reranker clearly improved ranking quality, particularly MRR.

However, retrieval metrics alone do not tell us whether users receive better answers.

For that reason, the experiment was extended to evaluate the complete RAG pipeline.

---

# Generation Evaluation

The retrieved documents are passed to an LLM to generate the final answer.

```text
Query
  ↓
Retrieval
  ↓
Context
  ↓
LLM
  ↓
Answer
```

Generation quality is evaluated across three dimensions.

### Correctness

Does the generated answer contain the essential information contained in the expected answer?

### Faithfulness

Are the claims made by the generated answer supported by the retrieved context?

### Answer Relevance

Does the answer directly address the user's question?

These metrics are intentionally evaluated separately.

For example, an answer can be faithful to the retrieved context but still be incorrect because retrieval failed to provide the necessary information.

---

# LLM-as-a-Judge

Generation evaluation is automated using an LLM-as-a-Judge.

```text
Question
Expected Answer
Retrieved Context
Generated Answer
        ↓
    LLM Judge
        ↓
Correctness
Faithfulness
Relevance
```

Each dimension uses a 0–2 rubric:

```text
0 = incorrect / criterion not satisfied
1 = partially correct / incomplete / minor issues
2 = correct and sufficiently complete
```

The scores are then normalized to a `0–1` scale.

LLM-as-a-Judge scores are treated as evaluation signals rather than absolute ground truth.

---

# End-to-End Results

The two strongest retrieval architectures were compared using the complete RAG pipeline.

| Metric | Dense | Dense + Reranker |
|---|---:|---:|
| Retrieval Recall@3 | 0.917 | **0.950** |
| Retrieval Precision@3 | 0.467 | 0.467 |
| Retrieval MRR | 0.789 | **0.900** |
| Generation Correctness | **0.867** | **0.867** |
| Generation Faithfulness | 0.967 | **1.000** |
| Generation Relevance | 0.967 | **1.000** |
| Retrieval Latency* | **~246 ms** | ~2207 ms |

\* Local benchmark. These latency measurements are environment-dependent and should not be interpreted as production infrastructure benchmarks.

---

# Architectural Decision

The Cross-Encoder significantly improved ranking quality:

```text
MRR
0.789 → 0.900
```

However, this improvement did not translate into higher end-to-end correctness:

```text
Correctness

Dense:             0.867
Dense + Reranker:  0.867
```

At the same time, the measured retrieval latency increased substantially in the local test environment.

For this dataset and experimental setup, the final architecture therefore uses:

```text
Dense Retrieval
      ↓
    Top 3
      ↓
     LLM
```

instead of always applying the Cross-Encoder.

The key engineering conclusion is:

> Improving an intermediate retrieval metric does not necessarily improve the final user-facing answer.

RAG architecture should therefore be evaluated end-to-end while considering quality, latency, infrastructure cost, and operational complexity.

This conclusion is specific to this dataset, models, hardware, and evaluation setup. Different applications may benefit significantly from reranking.

---

# Example: Diagnosing a Retrieval Failure

One evaluation question was:

```text
¿Qué diferencia hay entre valuation_details y risk_results?
```

The expected document was:

```text
architecture.txt
```

Dense Retrieval initially ranked it at position 4:

```text
1. incidents.txt
2. troubleshooting.txt
3. reconciliation.txt
4. architecture.txt
5. equity_method.txt
```

With `Top-K = 3`, the correct information therefore never reached the LLM.

The generated answer correctly reported that the available context was insufficient.

After Cross-Encoder reranking:

```text
1. reconciliation.txt
2. architecture.txt
3. equity_method.txt
...
```

The relevant document entered the final context and the LLM generated the correct answer.

This illustrates an important RAG debugging principle:

```text
Bad Answer
    ↓
Did the retrieved context contain the answer?
    │
    ├── NO  → Retrieval problem
    │
    └── YES → Generation / prompt / model problem
```

---

# Project Structure

```text
rag-evaluation-lab/
│
├── data/
│   └── docs/
│
├── src/
│   ├── document_loader.py
│   ├── ingest.py
│   ├── query.py
│   └── rag.py
│
├── evaluation/
│   ├── eval_dataset.json
│   ├── evaluate.py
│   ├── evaluate_generation.py
│   ├── inspect_scores.py
│   └── inspect_reranker.py
│
├── db/
│
├── .env
├── .gitignore
├── requirements.txt
├── main.py
└── README.md
```

---

# Tech Stack

- Python
- LangChain
- OpenAI API
- OpenAI Embeddings
- Chroma
- BM25
- Sentence Transformers
- BGE Cross-Encoder
- Pydantic

---

# Key Learnings

This project explores several practical RAG engineering concepts:

- Dense vs sparse retrieval
- Semantic vs lexical search
- Hybrid retrieval
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Multilingual retrieval considerations
- Recall@K
- Precision@K
- Mean Reciprocal Rank
- Golden datasets
- Retrieval evaluation
- Generation evaluation
- LLM-as-a-Judge
- Faithfulness
- Answer relevance
- Latency vs quality trade-offs
- End-to-end RAG evaluation

Most importantly, the project demonstrates that individual components should not be optimized in isolation.

A better retriever does not automatically produce a better RAG system.

---

# Limitations

This is an evaluation lab rather than a production RAG platform.

Current limitations include:

- Small synthetic corpus
- Only 15 evaluation queries
- File-level retrieval instead of chunk-level retrieval
- Local latency measurements
- LLM-as-a-Judge evaluation can be inconsistent
- No production-scale load testing
- No metadata filtering or access control
- No online evaluation or production observability

These limitations are intentional: the goal of the project is to understand and measure RAG architecture decisions in a controlled environment.

---

# Future Improvements

Potential extensions include:

- Chunk-level retrieval evaluation
- Metadata filtering
- Larger evaluation datasets
- Query rewriting
- Multi-query retrieval
- HyDE
- Conditional reranking
- Smaller/faster reranker models
- Retrieval and generation tracing
- Production observability
- Human evaluation alongside LLM-as-a-Judge
- Automated regression testing for RAG changes

---

## Author

Built as a practical AI Engineering project focused on Retrieval-Augmented Generation, evaluation, and evidence-based architecture decisions.