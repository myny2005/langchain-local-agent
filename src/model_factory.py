from settings import load_config
from dotenv import load_dotenv

load_dotenv()

# Load configuration
config = load_config()
provider = config["models"]["provider"]
models_cfg = config["models"][provider]  # Automatically picks the right group

# Initialize models
if provider == "openai":
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI

    embeddings = OpenAIEmbeddings(model=models_cfg["embeddings"])
    llm = ChatOpenAI(model=models_cfg["llm"], temperature=0.7)
elif provider == "ollama":
    from langchain_ollama import OllamaEmbeddings, ChatOllama

    embeddings = OllamaEmbeddings(model=models_cfg["embeddings"])  # type: ignore[assignment]
    llm = ChatOllama(model=models_cfg["llm"])  # type: ignore[assignment]
