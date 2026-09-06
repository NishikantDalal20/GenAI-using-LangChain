# LangChain 

## 1. What is LangChain?
- Open-source **framework for building LLM-based applications**.
- Connects components: LLMs, prompt templates, document loaders, text splitters, embedding models, vector databases, retrievers, tools, memory/state, workflows.
- **LangChain is NOT an LLM** — it's the framework used to build applications *around* LLMs.
- Manages the complexity of stitching together multiple components so developers don't rewrite orchestration boilerplate.
- Name comes from **chaining** tasks — output of one component feeds into the next.

## 2. Motivating Example: "Chat with PDF"
- User uploads a 1000-page PDF and asks e.g. *"What are the advantages of Linear Regression?"*
- Behind the scenes, the system must: store the PDF → load it → split into chunks → generate embeddings → store in a vector DB → embed the user's question → search for relevant chunks → send context + question to the LLM → generate the answer.
- Doing all this manually is complex — **LangChain orchestrates it.**

## 2a. Keyword Search vs. Semantic Search
| | Keyword Search | Semantic Search |
|---|---|---|
| How it works | Matches exact words | Matches by **meaning** of the query |
| Weakness | Too many/irrelevant results; no understanding of intent | — |
| Strength | — | Understands context; e.g. matches "benefits of linear regression" to a query about "advantages of linear regression" even without exact phrase overlap |

## 3. How Semantic Search Works — Embeddings
- An **embedding** converts text into a numerical vector representing its semantic meaning: `Text → Embedding Model → Vector [0.21, -0.43, 0.67, ...]`
- Both documents and the user's query are embedded into the same vector space.
- The system compares vectors using a similarity/distance measure (e.g., cosine similarity) — closest vectors = most relevant text.
- Flow: `Text → Embedding Model → Vector → Similarity Search → Relevant Text`

## 4. PDF Q&A System Architecture

**Ingestion pipeline:**
```
PDF Upload → Cloud Storage → Document Loader → Text Splitter → Embedding Model → Vector Database
```

**Query pipeline:**
```
User Query → Query Embedding → Vector Similarity Search → Relevant Chunks → Query + Context → LLM → Final Answer
```

### Why split documents into chunks?
- Large documents can't be efficiently processed as one giant block of text.
- Split by pages, paragraphs, chapters, fixed-size chunks, or other logical boundaries (e.g., a 1000-page PDF → 1000 chunks), each chunk then embedded.

### Vector database
- Stores all chunk embeddings.
- On a query: embed the question → compare to stored vectors → retrieve the most similar ones.
- Often configured with **Top-K** (e.g., Top-K = 5) to return the 5 most relevant chunks.

### The LLM as the "Brain"
Given `User Query + Relevant Context → LLM → Answer`, the LLM performs two jobs:
1. **Natural Language Understanding (NLU)** — understands the question's intent and meaning.
2. **Context-aware text generation** — uses retrieved context to generate the final answer.
- Modern LLMs already provide both, so developers don't need to build this "brain" from scratch.

### Why not feed the LLM the entire document?
Instead of `1000-page PDF → LLM`, retrieve only relevant chunks first: `PDF → Semantic Search → Relevant Chunks → LLM`
- **Benefits:** less computation, less irrelevant context, lower cost, faster processing, more focused answers.
- This retrieve-then-generate pattern is the core idea behind **RAG (Retrieval-Augmented Generation)**.

## 5. Three Major Challenges in Building LLM Apps

**Challenge 1 — Language understanding & generation**
- Historically hard to build a system that understands language and generates meaningful text.
- Solved by the evolution: `NLP → Transformers (2017) → BERT/GPT → Modern LLMs`. Developers now use existing LLMs instead of building their own.

**Challenge 2 — Computational cost**
- Hosting an LLM yourself needs GPUs, infra, engineering effort, high running cost.
- **Solution: LLM APIs** — `App → LLM API → Provider's server → Response` (e.g., OpenAI, Anthropic).
- Benefits: no self-hosting, easy integration, usage-based pricing, less infra to manage.

**Challenge 3 — Orchestration**
- Many moving parts: cloud storage, document loader, text splitter, embedding model, vector DB, retriever, LLM.
- Many tasks: load → split → embed → store → retrieve → call LLM → generate response.
- Manually wiring all this = lots of brittle boilerplate → **this is exactly what LangChain solves.**

## 6. Chains in LangChain
- A **chain** links multiple operations into a pipeline: `Input → Task 1 → Task 2 → Task 3 → Task 4 → Output`.
- One component's output becomes the next one's input.
- LangChain supports sequential, parallel, and conditional workflows — chains are a core LangChain concept.

## 7. Model-Agnostic Development
- LangChain abstracts the LLM/embedding provider behind a common interface: `App → LLM Interface → Provider A` (swappable later for `Provider B`) without rewriting core logic.
- Lets developers focus on application/business logic rather than provider-specific code.

## 8. LangChain Ecosystem
| Component | Purpose |
|---|---|
| Document Loaders | Load data from PDFs, files, cloud storage, etc. |
| Text Splitters | Break large documents into smaller chunks |
| Embeddings | Convert text into vectors |
| Vector Databases | Store & search embeddings |
| LLMs | Understand and generate text |

## 9. Memory and State
- LLM apps often need to remember conversation context.
- Example: Q1 = "What are the advantages of Linear Regression?" → Q2 = "Give me some interview questions about **this**."
- The system must resolve "this" → Linear Regression. **Memory/state** components handle this.

## 10. What You Can Build with LangChain

**1. Conversational chatbots** — handle repetitive queries; escalate to humans when unable to resolve (`Customer → AI Chatbot → [if unsolved] → Human Support`).

**2. AI Knowledge Assistants** — connected to specific data sources: courses, company docs, internal knowledge bases, product documentation.

**3. AI Agents** — more capable than simple Q&A chatbots; they can **use tools and take actions**:
```
User Request → Understand → Plan → Use Tools → Perform Action → Result
```
Example: "Find and book the cheapest flight" — an agent can search, compare, and complete the booking.

**4. Workflow Automation** — automate multi-step business, data, document, or personal workflows.

**5. Summarization & Research Assistants** — process books, papers, and large/company documents to provide summaries, Q&A, and information extraction.

## 11. Alternatives to LangChain
- **LlamaIndex** — popular for building RAG-style apps over external/private data.
- **Haystack** — open-source framework for LLM, search, and RAG applications.
- Best choice depends on project requirements, features, integrations, pricing, and developer preference.