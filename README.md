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

---

## 📚 Lecture 4 — LangChain Structured Output


### 1. What is Structured Output?
- **Unstructured output** — plain text, hard for programs to parse.
- **Structured output** — every field has a fixed structure (e.g., `time`, `activity`).
- **Definition:** asking an LLM to return responses in a well-defined data format (JSON, dictionary, object, etc.).

### 2. Why We Need It — Use Cases

**1. Resume → Database**
`Resume PDF → LLM extracts fields → JSON (name, marks, college...) → Insert into SQL`

**2. Review Analysis API**
Input: *"Battery is excellent but camera is average..."*
Output:
```json
{
  "topics": ["battery", "camera"],
  "pros": ["Excellent battery"],
  "cons": ["Average camera"],
  "sentiment": "Positive"
}
```
Perfect for Flask/FastAPI APIs.

**3. AI Agents ⭐**
User: *"Find square root of 2"* →
```json
{"operation": "sqrt", "number": 2}
```
The calculator tool can now execute it directly.
> **Key point:** tools understand structured data, not natural language.

### 3. How LangChain Creates Structured Output
Normal flow: `Prompt → Model.invoke() → Text response`
Structured flow: `Prompt → with_structured_output(schema) → JSON Output`

Only one extra step:
```python
structured_model = model.with_structured_output(schema)
result = structured_model.invoke(prompt)
```

### 4. Three Ways to Define a Schema
| Method | Best For |
|---|---|
| **TypedDict** | Simple Python projects |
| **Pydantic** | Validation + production |
| **JSON Schema** | Multi-language projects |

All three describe the same underlying structure in different ways.

---

### Part A — TypedDict
- Defines what keys and value types a dictionary should contain.
```python
class Person(TypedDict):
    name: str
    age: int
```
- **Limitation:** type hints only, **no runtime validation** — `{"name": "Nitish", "age": "25"}` (age as string) still runs without error.

**With LangChain:**
```python
class Review(TypedDict):
    summary: str
    sentiment: str
```
Result accessed like a dict: `result["summary"]`, `result["sentiment"]`.

**Adding descriptions (`Annotated`):**
```python
summary: Annotated[str, "Brief summary of the review"]
```
Descriptions help the LLM understand exactly what to generate.

**Advanced field types:**
- **List:** `themes: list[str]` → `["Battery", "Display", "Processor"]`
- **Optional:** `pros: Optional[list[str]]` — field omitted if unavailable
- **Literal (restrict values):** `sentiment: Literal["Positive", "Negative"]` — model can only choose these values

---

### Part B — Pydantic ⭐ (Recommended Approach)
- Unlike TypedDict, Pydantic performs **runtime validation**.
```python
class Student(BaseModel):
    name: str
```
- `{"name": "Nitish"}` ✅ valid; `{"name": 123}` ❌ raises a validation error.

**Key Pydantic features:**
1. **Default values** — `name: str = "Nitish"` (used if no value provided)
2. **Optional field** — `age: Optional[int] = None`
3. **Type coercion** — `"age": "32"` automatically becomes `age = 32`
4. **Email validation** — `email: EmailStr` accepts `abc@gmail.com`, rejects `abc`
5. **Field constraints** — `cgpa: float = Field(ge=0, le=10)` (8.5 ✅, 12 or -3 ❌)
6. **Description** — `summary: str = Field(description="Brief review summary")` — sent to the LLM too

**With LangChain:**
```python
class Review(BaseModel):
    themes: list[str]
    summary: str
    sentiment: Literal["Positive", "Negative"]
    pros: Optional[list[str]]
```
Result is a **Pydantic object** — accessed via `result.summary`, `result.sentiment` (not dict syntax).

---

### Part C — JSON Schema
- **Language-independent** schema format.
```json
{
  "title": "Review",
  "type": "object",
  "properties": {
    "summary": {"type": "string"}
  }
}
```
- **Use when:** Python backend + JavaScript frontend need to share the same schema across services.

---

### 5. TypedDict vs. Pydantic vs. JSON Schema
| Feature | TypedDict | Pydantic | JSON Schema |
|---|---|---|---|
| Type hints | ✅ | ✅ | ✅ |
| Runtime validation | ❌ | ✅ | ✅ |
| Default values | ❌ | ✅ | Limited |
| Optional fields | ✅ | ✅ | ✅ |
| Field descriptions | ✅ | ✅ | ✅ |
| Cross-language | ❌ | ❌ | ✅ |
| Best choice | Learning | **Production** | Multi-language |

> **Exam answer:** Pydantic is the best default choice for Python LangChain projects.

### 6. How `with_structured_output()` Works Internally
You write only:
```python
structured_model = model.with_structured_output(schema)
```
Behind the scenes, LangChain auto-generates a system prompt like:
> *"You are an AI assistant. Extract summary and sentiment. Return the response in JSON format."*

Then your actual prompt is appended.

**Internal flow:** `Auto System Prompt → User Review → LLM (structured generation) → JSON Output`

### 7. JSON Mode vs. Function Calling
`with_structured_output()` has a `method` parameter.

| Method | Purpose |
|---|---|
| `json_mode` | Return plain JSON |
| `function_calling` | Call tools/functions |

**JSON Mode** — best for APIs, databases, RAG, data extraction.
```json
{"summary": "..."}
```

**Function Calling** — best for AI Agents.
> User: *"Square root of 49"* →
```json
{"tool": "calculator", "operation": "sqrt", "number": 49}
```
The agent immediately calls the calculator.

**Rule of thumb:**
- OpenAI → Function Calling (default)
- Gemini/Claude → JSON Mode

### 8. Models That Support Structured Output
- **Supported:** GPT models, Claude, Gemini — use `with_structured_output()` directly.
- **Not supported:** TinyLlama, many Hugging Face models — need **Output Parsers** (covered in the next lesson).

---

## 📚 Lecture 5 — LangChain Output Parsers



### 1. What are Output Parsers?
> Output Parsers in LangChain convert raw LLM textual responses into structured formats like JSON, CSV, Pydantic models, etc.

**Benefits:** convert text → structured data · consistent output · validation support · easy integration with applications.

### 2. Four Important Output Parsers
1. **StrOutputParser** — returns only the string content
2. **JsonOutputParser** — returns a JSON dictionary
3. **StructuredOutputParser** — enforces a JSON schema
4. **PydanticOutputParser** — schema + validation

---

### 3. StrOutputParser
The simplest output parser — extracts only the text from the LLM response.

**Without parser:**
```python
result = model.invoke(prompt)
print(result.content)
```
**With parser:**
```python
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"topic": "Black Hole"})
```

**Why use it?** `model.invoke()` normally returns content + token usage + metadata + response id + finish reason. `StrOutputParser` strips everything except the text.

**Chain syntax (multi-step):**
```python
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Black Hole"})
```

**Best for:** chatbots, summarization, translation, multi-step chains.

---

### 4. JsonOutputParser
Forces the LLM to return JSON output.

```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()

template = PromptTemplate(
    template="""
    Give me the name, age and city of a fictional person.
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

**Why `get_format_instructions()`?** It automatically adds instructions like *"Return a JSON object"* so the LLM knows the expected format.

**Complete flow:**
```python
prompt = template.format()
result = model.invoke(prompt)
final = parser.parse(result.content)
```
**Output:** `{"name": "John", "age": 28, "city": "London"}` — Python type: `dict`.

**Chain version:**
```python
chain = template | model | parser
result = chain.invoke({})
```

**Limitation:** gives JSON, but does **not enforce a schema**. E.g. it might return `{"facts": ["...", "...", "..."]}` when you wanted `{"fact1": "...", "fact2": "...", "fact3": "..."}` — JsonOutputParser can't guarantee this exact structure.

---

### 5. StructuredOutputParser
Returns JSON according to a **predefined schema**.

| JSON Parser | Structured Parser |
|---|---|
| JSON only | JSON + Schema |
| No fixed keys | Fixed keys |
| Flexible | Controlled |

**Step 1 — Create response schema(s):**
```python
schemas = [
    ResponseSchema(name="fact1", description="First fact"),
    ResponseSchema(name="fact2", description="..."),
    ResponseSchema(name="fact3", description="...")
]
```

**Step 2 — Create parser:**
```python
parser = StructuredOutputParser.from_response_schemas(schemas)
```

**Step 3 — Prompt:**
```python
template = PromptTemplate(
    template="""
    Give 3 facts.
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

**Output:** `{"fact1": "....", "fact2": "....", "fact3": "...."}` — exactly follows the schema.

**Chain:**
```python
chain = template | model | parser
result = chain.invoke({"topic": "Black Hole"})
```

**Limitation:** schema (keys) is enforced, but **data validation is not**. E.g. if `age` should be `int` but the LLM returns `{"age": "35 years"}`, this parser accepts it anyway.

---

### 6. PydanticOutputParser
Uses Pydantic models to enforce **schema + data type + validation**. This is the **most powerful** parser discussed.

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
```

**Step 1 — Create Pydantic model:**
```python
class Person(BaseModel):
    name: str = Field(description="Person name")
    age: int = Field(gt=18, description="Age")
    city: str = Field(description="City")
```
`gt=18` means: age must be greater than 18.

**Step 2 — Parser:**
```python
parser = PydanticOutputParser(pydantic_object=Person)
```

**Step 3 — Prompt:**
```python
template = PromptTemplate(
    template="""
    Generate name, age and city.
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

**Output:** `Person(name='Rahul', age=24, city='Delhi')` — **not** a dictionary, a Pydantic object.

**Chain:**
```python
chain = template | model | parser
result = chain.invoke({"place": "Indian"})
```

---

### 7. Why Partial Variables?
Used for values that don't change at runtime — e.g. `format_instructions` (from the parser) is combined with runtime input (e.g. `topic = "Black Hole"`) into a single prompt.
```python
partial_variables={"format_instructions": parser.get_format_instructions()}
```

### 8. Chain Pipeline (General Syntax)
```
User Input → PromptTemplate (+ format instructions) → LLM Model (GPT/Gemini/Llama) → Output Parser (String/JSON/Structured/Pydantic) → Structured Output (JSON/Object)
```
```python
chain = prompt | model | parser
result = chain.invoke(input)
```
This `prompt | model | parser` syntax is the preferred LangChain style.

---

### 9. Comparison Table (Most Important)
| Feature | Str | JSON | Structured | Pydantic |
|---|---|---|---|---|
| Output type | String | JSON | JSON | Pydantic Object |
| Removes metadata | ✅ | ✅ | ✅ | ✅ |
| JSON format | ❌ | ✅ | ✅ | ✅ |
| Schema enforcement | ❌ | ❌ | ✅ | ✅ |
| Data validation | ❌ | ❌ | ❌ | ✅ |
| Best for | Text | JSON | Fixed JSON | Production Apps |

### 10. Which Parser Should You Use?
- Need plain text? → **StrOutputParser**
- Need JSON only? → **JsonOutputParser**
- Need fixed keys in JSON? → **StructuredOutputParser**
- Need schema + validation + production-ready output? → **PydanticOutputParser**

---
## 📚 Lecture 6 — LangChain Chains

### 1. What is a Chain?

A **Chain** is a pipeline that connects multiple LangChain components together.

```text
Input → Prompt → LLM → Parser → Output
```

The output of one component becomes the input of the next component.

Example:
```python
chain = prompt | model | parser
```

---

### 2. Why Chains?

Without Chains, each component must be invoked separately.

With Chains, the complete workflow can be executed using:

```python
result = chain.invoke(input)
```

#### Advantages

- Connects multiple components.
- Automates execution.
- Reduces manual code.
- Makes workflows easier to understand.
- Supports sequential, parallel, and conditional execution.

---

### 3. LCEL

The syntax used to connect LangChain components with the pipe operator is called:

> **LangChain Expression Language (LCEL)**

Example:
```python
chain = prompt | model | parser
```

Flow:

```text
Prompt
  ↓
Model
  ↓
Parser
  ↓
Output
```

LCEL is closely related to **Runnables**.

---

### 4. Invoking a Chain

Use `invoke()` to execute a Chain.

```python
result = chain.invoke({
    "topic": "cricket"
})

print(result)
```

To visualize the Chain:

```python
chain.get_graph().print_ascii()
```

---

### 5. Types of Chains

There are three important execution patterns:

```text
Sequential
Parallel
Conditional
```

---

#### 6. Sequential Chain

A Sequential Chain executes steps **one after another**.

```text
A → B → C → D
```

The output of one step becomes the input of the next.

#### Example

```text
Topic
 ↓
Detailed Report
 ↓
Five Point Summary
```

#### Code

```python
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a five pointer summary from:\n{text}",
    input_variables=["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser
```

Invoke:

```python
result = chain.invoke({
    "topic": "Unemployment in India"
})
```

---

### 7. Parallel Chain

A Parallel Chain executes **independent tasks at the same time**.

```text
             ┌→ Chain A
Input ───────┤
             └→ Chain B
```

Both Chains receive the same input.

### Example

From a document, generate:

- Notes
- Quiz

```text
              ┌→ Notes
Document ─────┤
              └→ Quiz
```

---

## RunnableParallel

`RunnableParallel` is used to execute multiple Runnables/Chains in parallel.

```python
from langchain_core.runnables import RunnableParallel

notes_chain = notes_prompt | model1 | parser
quiz_chain = quiz_prompt | model2 | parser

parallel_chain = RunnableParallel({
    "notes": notes_chain,
    "quiz": quiz_chain
})
```

The output has the structure:

```text
{
    "notes": ...,
    "quiz": ...
}
```

---

## Merge Parallel Outputs

The outputs can be passed to another Chain.

```python
merge_chain = merge_prompt | model | parser

chain = parallel_chain | merge_chain
```

Flow:

```text
             ┌→ Notes ──┐
Input ───────┤           ├→ Merge → Output
             └→ Quiz ───┘
```

---

### 8. Conditional Chain

A Conditional Chain selects a path based on a condition.

```text
Input
  ↓
Condition
 ┌┴──────┐
 ↓      ↓
A       B
```

Only the matching branch executes.

#### Example

Classify user feedback as:

```text
positive
negative
```

Then execute the appropriate Chain.

```text
Feedback
   ↓
Sentiment Classification
   ↓
Condition
  ↙     ↘
Positive Negative
  ↓       ↓
Chain    Chain
```

---

### 9. Structured Output for Classification

Normal string output may not always be consistent.

For example:

```text
positive
```

or:

```text
The sentiment is positive.
```

For reliable branching, we can use **PydanticOutputParser**.

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)
```

The result can be accessed using:

```python
result.sentiment
```

Example:

```python
result = classification_chain.invoke({
    "feedback": "This is a wonderful smartphone."
})

print(result.sentiment)
```

Output:

```text
positive
```

#### Format Instructions

```python
parser2.get_format_instructions()
```

These instructions can be passed to the prompt:

```python
prompt = PromptTemplate(
    template="""
    Classify the sentiment of the following feedback:

    {feedback}

    {format_instructions}
    """,
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)
```

Classification Chain:

```python
classification_chain = prompt | model | parser2
```

---

#### 10. RunnableBranch

`RunnableBranch` is used for **conditional execution**.

It works similar to:

```python
if
elif
else
```

#### Syntax

```python
from langchain_core.runnables import RunnableBranch

branch_chain = RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default_chain
)
```

#### Example

```python
positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        positive_chain
    ),
    (
        lambda x: x.sentiment == "negative",
        negative_chain
    ),
    default_chain
)
```

The complete workflow:

```python
chain = classification_chain | branch_chain
```

---

### 11. RunnableLambda

`RunnableLambda` converts a normal Python function or lambda into a Runnable.

```python
from langchain_core.runnables import RunnableLambda

default_chain = RunnableLambda(
    lambda x: "Could not find sentiment"
)
```

It is useful for creating custom logic inside a LangChain workflow.

---

### 12. Important LangChain Classes

| Class | Purpose |
|---|---|
| `PromptTemplate` | Creates dynamic prompts |
| `ChatOpenAI` | OpenAI chat model |
| `ChatAnthropic` | Anthropic chat model |
| `StrOutputParser` | Converts output to string |
| `PydanticOutputParser` | Creates structured Pydantic output |
| `RunnableParallel` | Executes multiple Runnables in parallel |
| `RunnableBranch` | Performs conditional execution |
| `RunnableLambda` | Converts Python functions into Runnables |

---

### 13. Important Syntax

#### Simple Chain

```python
chain = prompt | model | parser
```

#### Invoke

```python
chain.invoke(input)
```

#### Visualize

```python
chain.get_graph().print_ascii()
```

#### Sequential

```python
chain = prompt1 | model | parser | prompt2 | model | parser
```

#### Parallel

```python
RunnableParallel({
    "a": chain1,
    "b": chain2
})
```

#### Conditional

```python
RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default_chain
)
```

#### Runnable Function

```python
RunnableLambda(lambda x: ...)
```

---

### 14. Sequential vs Parallel vs Conditional

| Type | Use | Execution |
|---|---|---|
| **Sequential** | Dependent tasks | One after another |
| **Parallel** | Independent tasks | Simultaneously |
| **Conditional** | Decision-based tasks | Matching branch only |

#### Sequential

```text
A → B → C
```

#### Parallel

```text
       ┌→ B
A ─────┤
       └→ C
```

#### Conditional

```text
       ┌→ B
A → Condition
       └→ C
```

---
