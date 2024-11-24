import json
import os
from datetime import datetime
import uuid

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from datahub.emitter.mce_builder import make_dataset_urn, make_ml_model_urn, make_data_job_urn, make_data_flow_urn
from datahub.metadata.schema_classes import (
    MLModelPropertiesClass,
    MLModelKeyPropertiesClass,
    DataJobPropertiesClass,
    DataJobInputOutputClass,
    DataFlowPropertiesClass,
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
            model=llm,
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

        # Emit chain metadata (as DataFlow)
        chain_urn = llm_emitter.emit_chain(
            chain=chain,
            chain_type="RunnableSequence",
            upstream_urns=[model_urn, prompt_urn],
            config={
                "max_retries": 3,
                "verbose": False
            }
        )

        # Run the chain
        question = "What is the capital of France?"
        start_time = datetime.now()
        answer = chain.invoke({"question": question})
        end_time = datetime.now()

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer.content}")

        # Emit run metadata (as DataJob)
        run_urn = llm_emitter.emit_run(
            run_id=str(uuid.uuid4()),
            inputs={"question": question},
            outputs={"answer": answer.content},
            status="completed",
            start_time=start_time,
            end_time=end_time,
            upstream_urns=[chain_urn],
            metrics={
                "tokenUsage": {
                    "promptTokens": 0,
                    "completionTokens": 0,
                    "totalTokens": 0
                },
                "latency": (end_time - start_time).total_seconds(),
                "cost": 0
            }
        )

        print(f"\nMetadata URNs:")
        print(f"Model (MLModel): {model_urn}")
        print(f"Prompt (Dataset): {prompt_urn}")
        print(f"Chain (DataFlow): {chain_urn}")
        print(f"Run (DataJob): {run_urn}")

    except Exception as e:
        print(f"Error: {str(e)}")
        if not is_dry_run:
            raise

if __name__ == "__main__":
    run_example()
