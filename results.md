# Benchmark results

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
