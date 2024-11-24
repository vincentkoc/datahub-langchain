import json
import os
from datetime import datetime
import time

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI

from datahub.emitter.mce_builder import make_dataset_urn
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    StatusClass,
    BrowsePathsClass,
    DatasetSnapshotClass,
    MetadataChangeEventClass,
)
from src.metadata_setup import get_datahub_emitter

load_dotenv()

# Add delay between metadata pushes
METADATA_PUSH_DELAY = 2.0


class LangChainMetadataEmitter:
    def __init__(self, gms_server: str = None):
        # Default to GMS endpoint if not specified
        if not gms_server:
            gms_server = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
            # If using frontend, append /api/gms
            if ":9002" in gms_server:
                gms_server = f"{gms_server}/api/gms"

        self.emitter = get_datahub_emitter(gms_server)
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    def emit_metadata(self, dataset_urn: str, name: str, description: str, custom_properties: dict, browse_paths: list = None) -> str:
        """Emit metadata with consistent structure"""
        try:
            # Create browse paths
            browse_paths_aspect = BrowsePathsClass(
                paths=browse_paths or ["/langchain"]
            )

            # Create properties
            properties = DatasetPropertiesClass(
                name=name,
                description=description,
                customProperties=custom_properties
            )

            # Create status
            status = StatusClass(removed=False)

            # Create MetadataChangeEvent
            mce = MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=dataset_urn,
                    aspects=[
                        status,
                        properties,
                        browse_paths_aspect
                    ]
                )
            )

            # Debug output
            print(f"\nEmitting metadata for {name}")
            print(f"URN: {dataset_urn}")
            if self.is_dry_run:
                print("Metadata structure:")
                print(json.dumps(mce.to_obj(), indent=2))

            # Emit metadata
            self.emitter.emit(mce)
            print(f"Successfully emitted metadata for {name}")

            # Add delay after emission
            if not self.is_dry_run:
                time.sleep(METADATA_PUSH_DELAY)

            return dataset_urn

        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            print(f"Error: {error_msg}")
            if not self.is_dry_run:
                raise
            return None

    def emit_model_metadata(self, model):
        """Emit metadata for an LLM model"""
        model_urn = make_dataset_urn(
            platform="langchain",
            name=f"model_{model.model_name}",
            env="PROD"
        )

        custom_properties = {
            "provider": "OpenAI",
            "model_name": model.model_name,
            "parameters": json.dumps(model.model_kwargs),
            "modelFamily": "GPT",
            "modelType": "chat",
            "capabilities": json.dumps(["chat", "text-generation"]),
            "created_at": datetime.now().isoformat()
        }

        browse_paths = [
            "/langchain/models",
            f"/langchain/models/{model.model_name}"
        ]

        return self.emit_metadata(
            dataset_urn=model_urn,
            name=model.model_name,
            description=f"LLM Model: {model.model_name}",
            custom_properties=custom_properties,
            browse_paths=browse_paths
        )

    def emit_prompt_metadata(self, prompt):
        """Emit metadata for a prompt template"""
        # Format messages
        formatted_messages = []
        for message in prompt.messages:
            message_type = message.__class__.__name__.lower().replace(
                "messageprompttemplate", ""
            )
            formatted_messages.append({
                "role": message_type,
                "content": (
                    message.prompt.template
                    if hasattr(message.prompt, "template")
                    else str(message.prompt)
                ),
            })

        # Create stable ID for prompt
        prompt_str = json.dumps(formatted_messages, sort_keys=True)
        prompt_id = abs(hash(prompt_str))
        prompt_urn = make_dataset_urn(
            platform="langchain",
            name=f"prompt_{prompt_id}",
            env="PROD"
        )

        custom_properties = {
            "template": json.dumps(formatted_messages, indent=2),
            "inputVariables": json.dumps(list(prompt.input_variables)),
            "templateFormat": "chat",
            "category": "System",
            "version": "1.0",
            "tags": json.dumps(["chat", "system-prompt"]),
            "created_at": datetime.now().isoformat()
        }

        browse_paths = [
            "/langchain/prompts",
            f"/langchain/prompts/{prompt_id}"
        ]

        return self.emit_metadata(
            dataset_urn=prompt_urn,
            name=f"Prompt {prompt_id}",
            description="Chat prompt with system and human messages",
            custom_properties=custom_properties,
            browse_paths=browse_paths
        )

    def emit_chain_metadata(self, chain, model_urn, prompt_urn):
        """Emit metadata for an LLM chain"""
        chain_id = abs(hash(id(chain)))
        chain_urn = make_dataset_urn(
            platform="langchain",
            name=f"chain_{chain_id}",
            env="PROD"
        )

        custom_properties = {
            "chainType": "RunnableSequence",
            "components": json.dumps([model_urn, prompt_urn]),
            "category": "Generation",
            "configuration": json.dumps({
                "maxRetries": 3,
                "verbose": False,
            }),
            "inputSchema": json.dumps({
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }),
            "outputSchema": json.dumps({
                "type": "object",
                "properties": {"content": {"type": "string"}},
            }),
            "created_at": datetime.now().isoformat()
        }

        browse_paths = [
            "/langchain/chains",
            f"/langchain/chains/{chain_id}"
        ]

        return self.emit_metadata(
            dataset_urn=chain_urn,
            name=f"Chain {chain_id}",
            description="LangChain component for processing and generating text",
            custom_properties=custom_properties,
            browse_paths=browse_paths
        )


def run_example():
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
    if is_dry_run:
        print("\nRunning in DRY RUN mode - metadata will be printed but not sent to DataHub")

    # Initialize LangChain components
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])

    # Create chain
    chain = prompt | llm

    # Initialize metadata emitter
    metadata_emitter = LangChainMetadataEmitter()

    try:
        # Emit metadata for all components
        model_urn = metadata_emitter.emit_model_metadata(llm)
        prompt_urn = metadata_emitter.emit_prompt_metadata(prompt)
        chain_urn = metadata_emitter.emit_chain_metadata(chain, model_urn, prompt_urn)

        # Run the chain
        question = "What is the capital of France?"
        answer = chain.invoke({"question": question})
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer.content}")

        print(f"\nMetadata URNs:")
        print(f"Model: {model_urn}")
        print(f"Prompt: {prompt_urn}")
        print(f"Chain: {chain_urn}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        if not is_dry_run:
            raise


if __name__ == "__main__":
    run_example()
