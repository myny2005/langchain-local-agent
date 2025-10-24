from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from step4_qa import load_vectorstore, build_qa_chain
from settings import load_config
from model_factory import llm


def generate_one_question(vs, k_context: int) -> str:
    retriever = vs.as_retriever(search_kwargs={"k": k_context})
    sample_docs = retriever.invoke("global overview")
    sample_context = "\n\n".join(d.page_content for d in sample_docs)

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

    vs = load_vectorstore()
    qa = build_qa_chain(vs, k=cfg["retrieval"]["k"])

    question = generate_one_question(vs, k_context=cfg["generation"]["sample_k"])
    answer = qa.invoke(question).strip()

    print("\n❓ Auto-generated question:")
    print(question)
    print("\n💡 Answer:")
    print(answer)
