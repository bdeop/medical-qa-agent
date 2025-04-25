from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

def get_clarification_chain():
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    return ConversationChain(llm=llm, memory=memory)
