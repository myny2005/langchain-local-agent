from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from step4_qa import load_vectorstore, build_qa_chain
from settings import load_config


def generate_one_question(vs, model_name: str, k_context: int) -> str:
    retriever = vs.as_retriever(search_kwargs={"k": k_context})
    sample_docs = retriever.get_relevant_documents("global overview")
    sample_context = "\n\n".join(d.page_content for d in sample_docs)

    llm = ChatOllama(model=model_name, temperature=0.2)
    prompt = PromptTemplate(
        template=(
            "Propose exactly ONE high-impact question a researcher should ask about this article.\n"
            "Base it ONLY on the CONTEXT below. Avoid trivia; focus on implications or causal links.\n"
            "Return ONLY the question text, no quotes, no bullets, no JSON.\n\n"
            "CONTEXT:\n{context}\n\n"
            "Question:"
        ),
        input_variables=["context"],
    )
    chain = prompt | llm | StrOutputParser()
    q = chain.invoke({"context": sample_context}).strip()
    return q.strip("“”\"' ")


if __name__ == "__main__":
    cfg = load_config()
    model = cfg["models"]["llm"]

    vs = load_vectorstore()
    qa = build_qa_chain(vs, model_name=model, k=cfg["retrieval"]["k"])

    question = generate_one_question(
        vs, model_name=model, k_context=cfg["generation"]["sample_k"]
    )
    answer = qa.invoke(question).strip()

    print("\n❓ Auto-generated question:")
    print(question)
    print("\n💡 Answer:")
    print(answer)
