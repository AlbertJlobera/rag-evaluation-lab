## Experiment 1 — Top-K

| K | Recall@K | Precision@K | MRR |
|---|---:|---:|---:|
| 1 | 0.456 | 0.600 | 0.600 |
| 3 | 0.900 | 0.422 | 0.744 |
| 5 | 1.000 | 0.280 | 0.761 |

Increasing K improved recall but reduced precision.
K=5 retrieved all relevant documents in the evaluation dataset,
but introduced substantially more irrelevant context.

K=3 provides a better balance for the dense retrieval baseline.