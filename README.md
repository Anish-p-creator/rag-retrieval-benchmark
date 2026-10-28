# 🔬 RAG Retrieval Benchmark

A RAG system is only as good as what it retrieves: if the right chunk never reaches the prompt, the LLM can't use it. This project measures that directly by testing **chunking strategies × retrievers** on a labelled question set and reporting Recall@1, Recall@3 and MRR.

## What's compared

**Chunking strategies** (`chunking.py`)
- `fixed-300`: 300-character windows
- `fixed-600+100`: 600-character windows with 100 characters of overlap
- `sentences-2`: groups of two sentences
- `paragraph+title`: one chunk per paragraph, with the document title prepended

**Retrievers** (`retrievers.py`)
- **BM25**: keyword ranking with term-frequency saturation and length normalisation
- **TF-IDF**: cosine similarity over sublinear TF-IDF vectors
- **Dense**: all-MiniLM-L6-v2 sentence embeddings (when `sentence-transformers` is installed)
- **Hybrid RRF**: Reciprocal Rank Fusion of BM25 with Dense (or with TF-IDF in lexical-only mode)

## Dataset

`corpus.jsonl` holds 20 short technical documents on databases and infrastructure. Several are deliberately confusable: two-phase locking vs two-phase commit, write-ahead logging vs replication, Redis atomic counters vs Redis rate limiting, MVCC vs isolation levels, consistent hashing vs load balancing.

`queries.jsonl` holds 40 questions, each labelled with its gold document. Many are paraphrases that share few words with the answer ("make sure nothing committed is lost if the machine crashes"), which is where lexical retrievers start to struggle.

Metrics are document-level: a query counts as a hit if any retrieved chunk comes from the gold document.

## Results (lexical-only run)

`python benchmark.py --no-dense`

Corpus: 20 documents, 40 labelled queries

| Chunking | Chunks | Retriever | Recall@1 | Recall@3 | MRR |
|---|---:|---|---:|---:|---:|
| sentences-2 | 45 | Hybrid RRF (BM25 + TF-IDF) | 0.95 | 1.00 | 0.971 |
| sentences-2 | 45 | BM25 | 0.95 | 1.00 | 0.971 |
| fixed-300 | 41 | Hybrid RRF (BM25 + TF-IDF) | 0.95 | 1.00 | 0.967 |
| paragraph+title | 40 | Hybrid RRF (BM25 + TF-IDF) | 0.93 | 1.00 | 0.963 |
| fixed-600+100 | 21 | BM25 | 0.93 | 1.00 | 0.958 |
| paragraph+title | 40 | BM25 | 0.93 | 1.00 | 0.958 |
| fixed-600+100 | 21 | Hybrid RRF (BM25 + TF-IDF) | 0.93 | 0.97 | 0.955 |
| paragraph+title | 40 | TF-IDF | 0.93 | 0.97 | 0.953 |
| fixed-300 | 41 | BM25 | 0.93 | 0.95 | 0.950 |
| fixed-600+100 | 21 | TF-IDF | 0.93 | 0.97 | 0.949 |
| sentences-2 | 45 | TF-IDF | 0.90 | 0.97 | 0.936 |
| fixed-300 | 41 | TF-IDF | 0.90 | 0.95 | 0.934 |

**Takeaways from this run**
- Fusion is the most consistent option: hybrid RRF is at or near the top for every chunking strategy, because BM25 and TF-IDF miss different queries.
- Chunking matters about as much as the retriever. The same BM25 goes from 0.950 to 0.971 MRR just by changing how the text is split.
- Bigger chunks aren't automatically better: `fixed-600+100` produces half as many chunks but doesn't beat the smaller strategies.

Run without `--no-dense` to add the embedding retriever and the BM25 + Dense hybrid, which is where paraphrased queries should improve most.

## Run it

```bash
pip install -r requirements.txt
python benchmark.py --no-dense   # seconds, no model download
python benchmark.py              # adds Dense + BM25/Dense hybrid
```

Results print as a Markdown table and are saved to `results.md`. To benchmark your own data, point `--corpus` and `--queries` at JSONL files with the same fields.

## Stack

Python · rank-bm25 · scikit-learn · Sentence-Transformers · NumPy
