from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks
from omnibrain.agents.evaluators.grader import RetrievalGrader
from omnibrain.agents.nodes.query_rewriter import QueryRewriter

query = "What is RL?"

grader = RetrievalGrader()
rewriter = QueryRewriter()

print("FIRST SEARCH")
results1 = search_text_chunks(query)
grade1 = grader.grade(query, results1)
print(grade1)

if not grade1["is_relevant"]:
    print("\nRewriting query...")

    new_query = rewriter.rewrite(query, results1)

    print("New Query:", new_query)

    print("\nSECOND SEARCH")
    results2 = search_text_chunks(new_query)
    grade2 = grader.grade(new_query, results2)

    print(grade2)

    if grade2["confidence"] > grade1["confidence"]:
        print("\nImprovement achieved.")
    else:
        print("\nNo improvement.")
else:
    print("\nNo rewrite needed.")