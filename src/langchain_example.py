import json
import os
from datetime import datetime

from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI

from .metadata_setup import get_datahub_emitter

load_dotenv()


class LangChainMetadataEmitter:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    def emit_metadata(self, mce_dict: dict) -> str:
        """Emit metadata with dry run handling"""
        try:
            if self.is_dry_run:
                print(
                    f"\nDRY RUN: Would emit metadata for {mce_dict['proposedSnapshot']['urn']}"
                )
            self.emitter.emit(mce_dict)
            return mce_dict["proposedSnapshot"]["urn"]
        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            if self.is_dry_run:
                print(f"DRY RUN ERROR: {error_msg}")
            else:
                raise Exception(error_msg)
            return None

    def emit_model_metadata(self, model):
        """Emit metadata for an LLM model"""
        model_urn = f"urn:li:llmModel:{model.model_name}"

        mce_dict = {
            "proposedSnapshot": {
                "urn": model_urn,
                "aspects": [
                    {
                        "llmModelProperties": {
                            "modelName": model.model_name,
                            "provider": "OpenAI",
                            "parameters": model.model_kwargs,
                            "modelFamily": "GPT",
                            "modelType": "chat",
                            "capabilities": ["chat", "text-generation"],
                            "metrics": {
                                "averageLatency": None,
                                "tokenThroughput": None,
                                "errorRate": None,
                            },
                        }
                    }
                ],
            }
        }
        return self.emit_metadata(mce_dict)

    def emit_prompt_metadata(self, prompt):
        """Emit metadata for a prompt template"""
        # Properly extract message templates
        formatted_messages = []
        for message in prompt.messages:
            # Each message is a BaseMessagePromptTemplate
            # The type is determined by the class name
            message_type = message.__class__.__name__.lower().replace(
                "messageprompttemplate", ""
            )
            formatted_messages.append(
                {
                    "role": message_type,  # Will be 'system', 'human', etc.
                    "content": (
                        message.prompt.template
                        if hasattr(message.prompt, "template")
                        else str(message.prompt)
                    ),
                }
            )

        # Create a stable string for hashing
        prompt_str = json.dumps(formatted_messages, sort_keys=True)
        prompt_id = hash(prompt_str)
        prompt_urn = f"urn:li:llmPrompt:{prompt_id}"

        mce_dict = {
            "proposedSnapshot": {
                "urn": prompt_urn,
                "aspects": [
                    {
                        "llmPromptProperties": {
                            "template": json.dumps(formatted_messages, indent=2),
                            "inputVariables": list(prompt.input_variables),
                            "templateFormat": "chat",
                            "category": "System",
                            "metadata": {
                                "description": "Chat prompt with system and human messages",
                                "createdAt": datetime.now().isoformat(),
                                "usage": {
                                    "totalCalls": 0,
                                    "successRate": 0.0,
                                    "averageTokens": 0,
                                },
                            },
                            "version": "1.0",
                            "tags": ["chat", "system-prompt"],
                            "examples": [],
                        }
                    }
                ],
            }
        }
        return self.emit_metadata(mce_dict)

    def emit_chain_metadata(self, chain, model_urn, prompt_urn):
        """Emit metadata for an LLM chain"""
        chain_id = id(chain)
        chain_urn = f"urn:li:llmChain:{chain_id}"

        mce_dict = {
            "proposedSnapshot": {
                "urn": chain_urn,
                "aspects": [
                    {
                        "llmChainProperties": {
                            "chainType": "RunnableSequence",
                            "components": [model_urn, prompt_urn],
                            "description": "LangChain component for processing and generating text",
                            "category": "Generation",
                            "configuration": {
                                "maxRetries": 3,
                                "verbose": False,
                                "callbacks": [],
                            },
                            "performance": {
                                "averageLatency": None,
                                "successRate": None,
                                "costPerRun": None,
                            },
                            "inputSchema": {
                                "type": "object",
                                "properties": {"question": {"type": "string"}},
                                "required": ["question"],
                            },
                            "outputSchema": {
                                "type": "object",
                                "properties": {"content": {"type": "string"}},
                            },
                        }
                    }
                ],
            }
        }
        return self.emit_metadata(mce_dict)


def run_example():
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
    if is_dry_run:
        print(
            "\nRunning in DRY RUN mode - metadata will be printed but not sent to DataHub"
        )

    # Initialize LangChain components with chat model
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{question}")]
    )

    # Create chain using new style
    chain = prompt | llm

    # Initialize metadata emitter
    metadata_emitter = LangChainMetadataEmitter()

    try:
        # Emit metadata for all components
        model_urn = metadata_emitter.emit_model_metadata(llm)
        prompt_urn = metadata_emitter.emit_prompt_metadata(prompt)
        chain_urn = metadata_emitter.emit_chain_metadata(chain, model_urn, prompt_urn)

        # Run the chain (always run in both modes)
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
