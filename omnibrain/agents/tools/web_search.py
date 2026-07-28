from duckduckgo_search import DDGS


def execute_web_search(query: str, max_results: int = 3):
    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)

            for item in search_results:
                results.append({"text": item["body"], "source": item["href"]})

    except Exception as e:
        print(f"Web Search Error: {e}")

    return results
