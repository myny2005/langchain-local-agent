from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from settings import load_config, persist_path_for_url
from model_factory import llm, embeddings
import warnings

warnings.filterwarnings("ignore")


def load_vectorstore():
    cfg = load_config()
    url = cfg["article"]["url"]
    base_dir = cfg["index"]["base_dir"]
    persist_dir = persist_path_for_url(base_dir, url)

    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def build_qa_chain(vs, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful research assistant. Answer briefly and concisely, relying ONLY on the provided context. "
            "If the answer cannot be inferred from the context, reply with 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    chain = (
        {
            "context": retriever
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    cfg = load_config()
    vs = load_vectorstore()
    qa = build_qa_chain(vs, k=cfg["retrieval"]["k"])

    for q in ["what is the main conclusion of the article?"]:
        print(f"\n❓ {q}")
        print("💡", qa.invoke(q))
