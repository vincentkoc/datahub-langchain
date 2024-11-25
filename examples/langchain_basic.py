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
        # Setup observation with debug mode
        config = ObservabilityConfig(
            langchain_verbose=True,
            langchain_handler="default",
            datahub_dry_run=False  # Set to True to skip actual DataHub ingestion
        )

        # Initialize emitter with debug mode
        emitter = DataHubEmitter(
            gms_server="http://localhost:8080",
            debug=True
        )

        # Register platforms before using them
        try:
            emitter.register_platforms()
        except Exception as e:
            print(f"Warning: Platform registration failed: {e}")
            print("Continuing with execution...")

        # Create pipeline name
        pipeline_name = "langchain_basic"

        observer = LangChainObserver(
            config=config,
            emitter=emitter,
            pipeline_name=pipeline_name,
            group_models=False
        )

        # Initialize LangChain components
        llm = ChatOpenAI(
            callbacks=[observer],
            model_name="gpt-3.5-turbo"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])

        # Create and run chain
        chain = prompt | llm

        # Run with observation
        result = chain.invoke({"input": "Tell me a joke"})
        print(f"\nResponse: {result.content}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
