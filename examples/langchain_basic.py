import os
import sys
from pathlib import Path

# Add the src directory to the Python path for local development
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.platforms.langchain import LangChainObserver
from src.emitters.datahub import DataHubEmitter
from src.config import ObservabilityConfig

def main():
    # Load environment variables
    load_dotenv()

    # Verify OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    try:
        # Setup observation
        config = ObservabilityConfig(
            langchain_verbose=True,
            langchain_handler="default"
        )
        emitter = DataHubEmitter(
            gms_server="http://localhost:8080",
            debug=True
        )
        observer = LangChainObserver(
            config=config,
            emitter=emitter,
            pipeline_name="langchain_basic",
            group_models=False
        )

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

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
