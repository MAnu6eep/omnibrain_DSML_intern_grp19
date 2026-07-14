# omnibrain_DSML_intern_grp19
Bypassing the hallucination risks of standard LLMs

Agentic Multi-Modal RAG Orchestrator

## Team Members

- Anudeep
- Charan
- Om
- Manav
- Abhilash
- Meerja

## Initial understanding

- Why build another one
The reason is usually not “because no tool exists,” but because existing tools are often fragmented, generic, or not suited for a specific workflow. A new system becomes useful when it connects multiple tools into one coordinated pipeline, especially for tasks that need routing, reasoning, and combining results from different sources.

- What existing tools miss
Most tool directories are collections of separate utilities, not an orchestrated system that decides which tool to use, in what order, and how to merge the outputs. That matters when a problem needs several steps, like extracting tables, reading charts, querying text, and then producing one cited answer.

Why this approach still matters
In your OmniBrain-style setup, the value is not “another calculator” or “another parser.” The value is the supervisor layer that can coordinate specialized agents and reduce hallucinations by using the right retrieval method for each part of the document.

- Simple example
A normal tool site might give you one PDF converter, one chart reader, and one search box. An orchestrator can take a 500-page report, send tables to one agent, text to another, and stock history to a third, then synthesize a final memo automatically.
