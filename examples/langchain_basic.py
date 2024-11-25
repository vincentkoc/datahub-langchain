from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llm_observability.platforms.langchain import LangChainObserver
from llm_observability.emitters.datahub import DataHubEmitter

def main():
    # Setup observation
    emitter = DataHubEmitter()
    observer = LangChainObserver(emitter)

    # Initialize LangChain components
    llm = ChatOpenAI(
        callbacks=[observer]  # Add observer
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])

    # Create and run chain
    chain = prompt | llm

    # Run with observation
    result = chain.invoke({"input": "Tell me a joke"})
    print(result)

if __name__ == "__main__":
    main()
