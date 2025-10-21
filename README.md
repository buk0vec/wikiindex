# WikiIndex

RAG explorer created for my Cal Poly senior project. Includes:

- A server which downloads a Wikipedia HTML dump, parses and splits the article abstracts, and creates and embedding index with ColBERT before serving them.
- A server which serves different types of RAG pipelines using dspy, with some hacked-in support for SSE while the pipelines are executing.
- A frontend which allows users to log in and ask questions to each of the RAG pipelines and inspect their outputs across each step.
