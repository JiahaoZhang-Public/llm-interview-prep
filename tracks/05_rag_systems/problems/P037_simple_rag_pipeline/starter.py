def run_simple_rag(query: str, retriever, llm, prompt_template: str):
    results = retriever.search(query, top_k=3)
    context = "\n".join(r["text"] for r in results)
    prompt = prompt_template.format(query=query, context=context)
    return llm(prompt)
