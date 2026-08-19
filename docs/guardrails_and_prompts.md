# OmniBrain Guardrails & System Prompts Architecture

## 1. System Prompt Architecture

OmniBrain utilizes a multi-agent orchestration architecture where specialized agents process queries routed by a central Supervisor agent.

### Supervisor Router Prompt (`SUPERVISOR_PROMPT`)
The Supervisor node evaluates user query intent and routes to the appropriate agent:
- **`text_agent`**: For document text queries, PDF chapter summaries, and vector context retrieval.
- **`vision_agent`**: For visual elements, charts, diagrams, and image cropping requests.
- **`sql_agent`**: For structured database queries against the SQLite financial stock database.
- **`web_agent`**: For real-time web lookups via DuckDuckGo fallback.

### Synthesizer / Generator Prompt
The Generator synthesizes final grounded answers using retrieved text and image context. It strictly enforces:
- 100% factual grounding against retrieved document chunks.
- Inline citation format: `[Source: document.pdf, Page: N, Chunk: ID]`.
- Zero metric hallucination.

---

## 2. NeMo Guardrails Configuration Specs

OmniBrain integrates NeMo Guardrails with regex pattern fallbacks to sanitize inputs and outputs.

### Input Guardrails (`check_input`)
Filters user prompts before execution:
- **Safety Checks**: Blocks jailbreaks, prompt injection, harmful instructions, and security bypass attempts.
- **Scope Boundary Checks**: Restricts queries to the domain of PDF documents, financial analysis, and database metrics.

### Output Guardrails (`check_output`)
Validates model outputs before sending responses:
- **Grounding Verification**: Ensures generated content adheres to retrieved facts.
- **Harmful Content Filtering**: Rejects responses containing unauthorized code execution or unsafe content.

### CoLang Rails Specs (`omnibrain/config/rails.colang`)
Defines flow control rules for user intent and bot response policies:
```colang
define user ask off topic
  "How do I hack a server?"
  "Write malware code"

define bot respond off topic
  "I can only answer questions directly grounded in the provided PDF documents or database."

define flow off topic query
  user ask off topic
  bot respond off topic
```
