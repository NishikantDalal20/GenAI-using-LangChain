# LangChain 

## 📚 Lecture 1 — Introduction to LangChain
### 1. What is LangChain?
- Open-source **framework for building LLM-based applications**.
- Connects components: LLMs, prompt templates, document loaders, text splitters, embedding models, vector databases, retrievers, tools, memory/state, workflows.
- **LangChain is NOT an LLM** — it's the framework used to build applications *around* LLMs.
- Manages the complexity of stitching together multiple components so developers don't rewrite orchestration boilerplate.
- Name comes from **chaining** tasks — output of one component feeds into the next.

### 2. Motivating Example: "Chat with PDF"
- User uploads a 1000-page PDF and asks e.g. *"What are the advantages of Linear Regression?"*
- Behind the scenes, the system must: store the PDF → load it → split into chunks → generate embeddings → store in a vector DB → embed the user's question → search for relevant chunks → send context + question to the LLM → generate the answer.
- Doing all this manually is complex — **LangChain orchestrates it.**

### 2a. Keyword Search vs. Semantic Search
| | Keyword Search | Semantic Search |
|---|---|---|
| How it works | Matches exact words | Matches by **meaning** of the query |
| Weakness | Too many/irrelevant results; no understanding of intent | — |
| Strength | — | Understands context; e.g. matches "benefits of linear regression" to a query about "advantages of linear regression" even without exact phrase overlap |

### 3. How Semantic Search Works — Embeddings
- An **embedding** converts text into a numerical vector representing its semantic meaning: `Text → Embedding Model → Vector [0.21, -0.43, 0.67, ...]`
- Both documents and the user's query are embedded into the same vector space.
- The system compares vectors using a similarity/distance measure (e.g., cosine similarity) — closest vectors = most relevant text.
- Flow: `Text → Embedding Model → Vector → Similarity Search → Relevant Text`

### 4. PDF Q&A System Architecture

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

### 5. Three Major Challenges in Building LLM Apps

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

### 6. Chains in LangChain
- A **chain** links multiple operations into a pipeline: `Input → Task 1 → Task 2 → Task 3 → Task 4 → Output`.
- One component's output becomes the next one's input.
- LangChain supports sequential, parallel, and conditional workflows — chains are a core LangChain concept.

### 7. Model-Agnostic Development
- LangChain abstracts the LLM/embedding provider behind a common interface: `App → LLM Interface → Provider A` (swappable later for `Provider B`) without rewriting core logic.
- Lets developers focus on application/business logic rather than provider-specific code.

### 8. LangChain Ecosystem
| Component | Purpose |
|---|---|
| Document Loaders | Load data from PDFs, files, cloud storage, etc. |
| Text Splitters | Break large documents into smaller chunks |
| Embeddings | Convert text into vectors |
| Vector Databases | Store & search embeddings |
| LLMs | Understand and generate text |

### 9. Memory and State
- LLM apps often need to remember conversation context.
- Example: Q1 = "What are the advantages of Linear Regression?" → Q2 = "Give me some interview questions about **this**."
- The system must resolve "this" → Linear Regression. **Memory/state** components handle this.

### 10. What You Can Build with LangChain

**1. Conversational chatbots** — handle repetitive queries; escalate to humans when unable to resolve (`Customer → AI Chatbot → [if unsolved] → Human Support`).

**2. AI Knowledge Assistants** — connected to specific data sources: courses, company docs, internal knowledge bases, product documentation.

**3. AI Agents** — more capable than simple Q&A chatbots; they can **use tools and take actions**:
```
User Request → Understand → Plan → Use Tools → Perform Action → Result
```
Example: "Find and book the cheapest flight" — an agent can search, compare, and complete the booking.

**4. Workflow Automation** — automate multi-step business, data, document, or personal workflows.

**5. Summarization & Research Assistants** — process books, papers, and large/company documents to provide summaries, Q&A, and information extraction.

### 11. Alternatives to LangChain
- **LlamaIndex** — popular for building RAG-style apps over external/private data.
- **Haystack** — open-source framework for LLM, search, and RAG applications.
- Best choice depends on project requirements, features, integrations, pricing, and developer preference.

---

## 📚 Lecture 2 — Core LangChain Components

### The 6 Components at a Glance

| Component | Purpose |
|---|---|
| **Models** | Connect to LLMs & embedding models |
| **Prompts** | Create dynamic and structured inputs |
| **Chains** | Build multi-step pipelines |
| **Indexes** | Connect LLMs to PDFs, websites & databases |
| **Memory** | Remember previous conversations |
| **Agents** | Let the LLM choose tools & take actions |

---

### 1. Models (Most Important Component)
A **Model** is LangChain's standardized interface for talking to AI models.

- **Without LangChain:** OpenAI, Claude, and Gemini APIs each use different syntax.
- **With LangChain:** nearly the same code works across providers — switching models often needs only 1–2 line changes.

**Two types:**
- **Language Models** — Text in → Text out. Used for chatbots, assistants, agents.
- **Embedding Models** — Text in → Vector out. Used for semantic search and RAG.

### 2. Prompts
A **prompt** is the input sent to an LLM. Small wording changes can produce very different outputs — this is why prompt engineering matters.

**Prompt types:**
- **Dynamic prompts** — use placeholders like `{topic}` and `{tone}`.
- **Role prompts** — e.g. "You are an experienced doctor..."
- **Few-shot prompts** — give examples before asking the new question.

Example:
> System: "You are an experienced teacher."
> User: "Explain gravity in a fun tone."

### 3. Chains
**Chains** connect multiple steps into a pipeline, where each component's output automatically becomes the next component's input.

Example workflow: `English Text → Translate to Hindi → Summarize (<100 words)`

**Chain types:**
- **Sequential chains** — step-by-step.
- **Parallel chains** — multiple LLMs work simultaneously.
- **Conditional chains** — different paths based on conditions.

### 4. Indexes (RAG Foundation)
**Indexes** let an LLM use external knowledge — PDFs, websites, or company databases.

**Pipeline:** `PDF/Website/DB → Loader → Splitter (Chunks) → Vector Store (Embeddings) → Retriever (Relevant Chunks)`

**4 sub-components:**
1. **Document Loader** — loads files.
2. **Text Splitter** — breaks content into chunks.
3. **Vector Store** — stores embeddings.
4. **Retriever** — fetches relevant chunks via semantic search.

This is the foundation of **RAG (Retrieval-Augmented Generation)**.

### 5. Memory
LLM APIs are **stateless** — they don't remember prior messages unless the full history is resent. **Memory** solves this.

| Memory Type | Keeps |
|---|---|
| Buffer | Entire conversation |
| Buffer Window | Last N messages |
| Summary Memory | Summary of the chat |
| Custom Memory | User preferences & facts |

Example:
> User: "Who is Narendra Modi?"
> User: "How old is he?"

Without memory, the model may not resolve "he" back to Narendra Modi.

### 6. Agents (Most Advanced Component)
Instead of following a fixed pipeline, an **Agent** decides *which tool to use* based on the user's request — the LLM becomes a decision-maker, not just a text generator.

Example: *"Summarize this PDF and email it."*
**Agent reasoning:** Read PDF → Summarize → Use email tool → Send result.

---

### Overall Architecture
```
User Input
   ↓
Prompts (Dynamic • Role • Few-shot)
   ↓
Models (LLMs & Embeddings) ←── Indexes (RAG)
   ↓
Chains (Pipeline orchestration) ←── Memory (Context)
   ↓
Agents (Reason + Tools + Actions)
   ↓
Final Response
```

---

## 📚 Lecture 3 — LangChain Prompts

### 1. What is a Prompt?
- The message sent to an LLM. Even small wording changes → very different outputs → **prompt engineering matters**.
- Example: `"Write a 5-line poem on cricket."`

**Prompt types:**
| Type | Example |
|---|---|
| Text | Ask a question |
| Image | Upload image + ask |
| Audio | Upload song |
| Video | Upload video |

This lesson focuses only on **text prompts** (most common in LangChain today).

### 2. Static vs. Dynamic Prompts

**Static Prompt** — user writes the entire instruction manually.
- Problem: different users write different prompts → inconsistent outputs, spelling mistakes, missing details, inconsistent style.

**Dynamic Prompt** — the app fills a template from structured inputs (e.g., dropdowns).
- Benefits: consistent responses, controlled user input, better UX, less hallucination from poorly-written prompts.
- **Preferred approach for real applications.**

### 3. PromptTemplate
LangChain's `PromptTemplate` builds dynamic prompts using placeholders.

```python
template = PromptTemplate(
    template="Summarize {paper} in {style} style",
    input_variables=["paper", "style"]
)
```
Later:
```python
prompt = template.invoke({
    "paper": "Attention Is All You Need",
    "style": "Simple"
})
```
**Flow:** `User Inputs → Prompt Template (fill placeholders) → Final Prompt`

### 4. Why Not Just Use f-strings?
Common interview question.

| Feature | PromptTemplate | f-string |
|---|---|---|
| Placeholder validation | ✅ | ❌ |
| Reusable | ✅ | Limited |
| Save/Load as JSON | ✅ | ❌ |
| Works with Chains | ✅ | ❌ |
| LangChain native | ✅ | ❌ |

**Three major advantages:**
1. **Validation** — checks every placeholder has a value; missing ones (e.g. `{length}`) throw an error before runtime.
2. **Reusability** — templates can be saved as JSON (`template.json`) and reloaded anywhere in the project.
3. **Chain compatibility** — templates connect directly into LangChain Chains for cleaner pipelines.

### 5. Example App: Research Assistant (Streamlit)
- User selects: **research paper**, **explanation style**, **summary length** (via dropdowns).
- These values auto-populate the prompt template — no manual typing needed.

### 6. Prompt → Model → Chain

**Without a Chain:** `Template.invoke() → Model.invoke() → Output` — **two** `invoke()` calls needed.

**With a Chain:** `Chain (Template → Model).invoke() → Result` — only **one** `invoke()` call, since the template and model are combined into a single pipeline.

### 7. Building a Chatbot — The Context Problem
- Basic chatbot: each message handled independently.
- Problem example:
  > User: "Which number is bigger: 2 or 0?" → AI: "2"
  > User: "Multiply the bigger number by 10."
  Without history, the AI can't resolve "the bigger number" → 2.

### 8. Chat History — the Fix
- Instead of sending just the current message, send: **previous messages + current message**.
- Now the AI can correctly resolve references like "the bigger number" or "this."

### 9. LangChain Message Types ⭐ (Key Interview Topic)
LangChain represents conversations using three message classes:

| Message | Purpose | Example |
|---|---|---|
| **SystemMessage** | Defines AI behavior before the conversation starts | "You are a helpful AI assistant." |
| **HumanMessage** | Message sent by the user | "Explain LangChain." |
| **AIMessage** | Response generated by the model | "LangChain is an orchestration framework..." |

Using these structured message objects (instead of plain strings) means the model always knows **who said what**.

### 10. Final Architecture
```
User Input
   ↓
PromptTemplate (fill placeholders)
   ↓
Messages (System • Human • AI)
   ↓
Chat Model — invoke()
   ↓
AI Response (stored as AIMessage)
```

