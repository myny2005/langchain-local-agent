from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

messages = [
	SystemMessage("You are an expert about football (soccer)"),
	HumanMessage("Give me 5 names of the best goalkeepers of all time")
]

result = llm.invoke(messages)

print(result.content)
