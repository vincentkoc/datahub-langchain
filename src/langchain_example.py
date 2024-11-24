import json
import os
from datetime import datetime
import uuid

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from datahub.emitter.mce_builder import make_dataset_urn, make_ml_model_urn
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    MLModelPropertiesClass,
    StatusClass,
    BrowsePathsClass,
)
from src.llm_metadata import LLMMetadataEmitter

load_dotenv()

def run_example():
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
    if is_dry_run:
        print("\nRunning in DRY RUN mode")

    # Initialize components
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    chain = prompt | llm

    # Initialize metadata emitter
    llm_emitter = LLMMetadataEmitter()

    try:
        # Emit model metadata (as MLModel)
        model_urn = llm_emitter.emit_model(
            model_name=llm.model_name,
            provider="OpenAI",
            model_type="chat",
            capabilities=["text-generation", "chat"],
            parameters={
                "contextWindow": 4096,
                "tokenLimit": 4096,
                "costPerToken": 0.0001
            }
        )

        # Emit prompt metadata (as Dataset)
        prompt_urn = llm_emitter.emit_prompt(
            prompt=prompt,
            template_format="chat",
            category="System",
            version="1.0"
        )

        # Emit chain metadata (as Dataset with pipeline subtype)
        chain_urn = llm_emitter.emit_pipeline(
            chain_type="RunnableSequence",
            config={
                "max_retries": 3,
                "verbose": False,
                "components": ["prompt", "model"]
            },
            upstream_urns=[model_urn, prompt_urn]
        )

        # Run the chain
        question = "What is the capital of France?"
        start_time = datetime.now()
        answer = chain.invoke({"question": question})
        end_time = datetime.now()

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer.content}")

        # Emit run metadata (as Dataset with run subtype)
        run_urn = llm_emitter.emit_run(
            run_id=str(uuid.uuid4()),
            inputs={"question": question},
            outputs={"answer": answer.content},
            status="completed",
            metrics={
                "tokenUsage": {
                    "promptTokens": 0,
                    "completionTokens": 0,
                    "totalTokens": 0
                },
                "latency": (end_time - start_time).total_seconds(),
                "cost": 0
            },
            upstream_urns=[chain_urn],
            tags=["example", "chat"]
        )

        # Emit lineage relationships
        llm_emitter.emit_lineage(chain_urn, model_urn, "Uses")
        llm_emitter.emit_lineage(chain_urn, prompt_urn, "Uses")
        llm_emitter.emit_lineage(run_urn, chain_urn, "ExecutedBy")

        print(f"\nMetadata URNs:")
        print(f"Model: {model_urn}")
        print(f"Prompt: {prompt_urn}")
        print(f"Chain: {chain_urn}")
        print(f"Run: {run_urn}")

    except Exception as e:
        print(f"Error: {str(e)}")
        if not is_dry_run:
            raise

if __name__ == "__main__":
    run_example()
