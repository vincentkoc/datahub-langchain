import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from metadata_setup import get_datahub_emitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent

load_dotenv()

class LangChainMetadataEmitter:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    def emit_metadata(self, mce: MetadataChangeEvent) -> str:
        """Emit metadata with dry run handling"""
        try:
            if self.is_dry_run:
                print(f"\nDRY RUN: Would emit metadata for {mce['proposedSnapshot']['urn']}")
                self.emitter.emit(mce)  # DryRunEmitter will handle printing
            else:
                self.emitter.emit(mce)
            return mce['proposedSnapshot']['urn']
        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            if self.is_dry_run:
                print(f"DRY RUN ERROR: {error_msg}")
            else:
                raise Exception(error_msg)

    def emit_model_metadata(self, model):
        """Emit metadata for an LLM model"""
        model_urn = f"urn:li:llmModel:{model.model_name}"

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "urn": model_urn,
                "aspects": [{
                    "llmModelProperties": {
                        "modelName": model.model_name,
                        "provider": "OpenAI",
                        "parameters": model.model_kwargs,
                        "modelFamily": "GPT"
                    }
                }]
            }
        )
        return self.emit_metadata(mce)

    def emit_prompt_metadata(self, prompt):
        """Emit metadata for a prompt template"""
        prompt_id = hash(prompt.template)  # Simple way to generate unique ID
        prompt_urn = f"urn:li:llmPrompt:{prompt_id}"

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "urn": prompt_urn,
                "aspects": [{
                    "llmPromptProperties": {
                        "template": prompt.template,
                        "inputVariables": prompt.input_variables,
                        "templateFormat": "f-string"
                    }
                }]
            }
        )
        return self.emit_metadata(mce)

    def emit_chain_metadata(self, chain, model_urn, prompt_urn):
        """Emit metadata for an LLM chain"""
        chain_id = id(chain)  # Use object ID as unique identifier
        chain_urn = f"urn:li:llmChain:{chain_id}"
        chain_type = chain.__class__.__name__

        # Get chain category based on type
        category_mapping = {
            "LLMChain": "Generation",
            "ConversationChain": "QA",
            "RetrievalQA": "QA",
            "ConversationalRetrievalChain": "QA",
            "SQLDatabaseChain": "Analysis",
            "APIChain": "Analysis",
            "TransformChain": "Generation",
            "SequentialChain": "Custom",
            "RouterChain": "Custom",
            "VectorDBQA": "QA",
            "PALChain": "Analysis",
            "QAWithSourcesChain": "QA",
            "GraphQAChain": "Analysis",
            "HypotheticalDocumentEmbedder": "Generation",
            "LLMMathChain": "Analysis",
            "OpenAIFunctions": "Generation"
        }

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "urn": chain_urn,
                "aspects": [{
                    "llmChainProperties": {
                        "chainType": chain_type,
                        "components": [model_urn, prompt_urn],
                        "description": "LangChain component for processing and generating text",
                        "icon": self.get_chain_icon(chain_type),
                        "category": category_mapping.get(chain_type, "Custom"),
                        "configuration": {
                            "maxRetries": getattr(chain, "max_retries", 3),
                            "verbose": getattr(chain, "verbose", False),
                            "callbacks": [cb.__class__.__name__ for cb in (getattr(chain, "callbacks", []) or [])]
                        },
                        "performance": {
                            "averageLatency": None,  # To be populated from metrics if available
                            "successRate": None,
                            "costPerRun": None
                        }
                    }
                }]
            }
        )
        return self.emit_metadata(mce)

    def get_provider_icon(self, provider):
        """Get the icon for a provider"""
        icons = {
            "OpenAI": "data:image/svg+xml;base64,<base64_encoded_openai_icon>",
            "Anthropic": "data:image/svg+xml;base64,<base64_encoded_anthropic_icon>",
            # Add more provider icons
        }
        return icons.get(provider, "")

    def get_chain_icon(self, chain_type):
        """Get the icon for a chain type"""
        # Base64 encoded SVG icons
        LANGCHAIN_ICON = """
        PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMjQwIDQxIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNjEuNTEzOSAxMS4xNTY5QzYwLjQ1MjcgMTEuMTU2OSA1OS40NTQ5IDExLjU2OCA1OC43MDggMTIuMzE0OEw1NS42ODk5IDE1LjMyNDhDNTQuODc1NyAxNi4xMzY4IDU0LjQ1NzQgMTcuMjY0MyA1NC41NDMxIDE4LjQyMDJDNTQuNTQ5MiAxOC40ODMzIDU0LjU1NTMgMTguNTQ2NCA1NC41NjE1IDE4LjYxMTVDNTQuNjY5NiAxOS40OTg4IDU1LjA1OTQgMjAuMjk4NiA1NS42ODk5IDIwLjkyNTRDNTYuMTI0NiAyMS4zNTg5IDU2LjYwNDEgMjEuNjMzNyA1Ny4xODU3IDIxLjgyNUM1Ny4yMTYzIDIyIDU3LjIzMjYgMjIuMTc3IDU3LjIzMjYgMjIuMzU0MUM1Ny4yMzI2IDIzLjE1MTkgNTYuOTIyNSAyMy45MDA4IDU2LjM1OTIgMjQuNDYyNUw1Ni4xNzM1IDI0LjY0NzdDNTUuMTY1NSAyNC4zMDM3IDU0LjMyNDcgMjMuODAxMSA1My41NjU2IDIzLjA0NEM1Mi41NTc2IDIyLjAzODYgNTEuODkwMyAyMC43Njg3IDUxLjYzOTMgMTkuMzc0N0w1MS42MDQ2IDE5LjE4MTNMNTEuNDUxNSAxOS4zMDU1QzUxLjM0NzUgMTkuMzg4OSA1MS4yNDk1IDE5LjQ3ODUgNTEuMTU3NyAxOS41N0w0OC4xMzk2IDIyLjU4QzQ2LjU5MjggMjQuMTIyNiA0Ni41OTI4IDI2LjYzNiA0OC4xMzk2IDI4LjE3ODZDNDguOTEzIDI4Ljk0OTkgNDkuOTI5MiAyOS4zMzY2IDUwLjk0NzUgMjkuMzM2NkM1MS45NjU4IDI5LjMzNjYgNTIuOTggMjguOTQ5OSA1My43NTM0IDI4LjE3ODZMNTYuNzcxNSAyNS4xNjg3QzU4LjMxODMgMjMuNjI2IDU4LjMxODMgMjEuMTE0NyA1Ni43NzE1IDE5LjU3QzU2LjM1OTIgMTkuMTU5IDU1Ljg2NzUgMTguODQ5NiA1NS4zMTA0IDE4LjY1MDJDNTUuMjc5OCAxOC40NjkgNTUuMjYzNCAxOC4yODc5IDU1LjI2MzQgMTguMTEwOUM1NS4yNjM0IDE3LjI0MzkgNTUuNjA2MyAxNi40MjE3IDU2LjIzNDggMTUuNzk0OUM1Ny4yNDQ5IDE2LjEzODggNTguMTQwNyAxNi42OTY1IDU4Ljg5NzggMTcuNDUxNUM1OS45MDM4IDE4LjQ1NDggNjAuNTY5MSAxOS43MjI3IDYwLjgyNDEgMjEuMTIwOEw2MC44NTg4IDIxLjMxNDFMNjEuMDExOSAyMS4xOUM2MS4xMTYgMjEuMTA2NiA2MS4yMTM5IDIxLjAxNyA2MS4zMDc4IDIwLjkyMzRMNjQuMzI1OSAxNy45MTM1QzY1Ljg3MjcgMTYuMzcwOCA2NS44NzQ3IDEzLjg1NzUgNjQuMzI1OSAxMi4zMTQ4QzYzLjU3NyAxMS41NjggNjIuNTgxMSAxMS4xNTY5IDYxLjUxOCAxMS4xNTY5SDYxLjUxMzlaIiBmaWxsPSJDdXJyZW50Q29sb3IiPjwvcGF0aD4KPHBhdGggZD0iTTU5Ljg5NjYgMC4xNDg4NjVIMjAuNDA2M0M5LjE1NDI2IDAuMTQ4ODY1IDAgOS4yNzg0MSAwIDIwLjUwMDFDMCAzMS43MjE3IDkuMTU0MjYgNDAuODUxMyAyMC40MDYzIDQwLjg1MTNINTkuODk2NkM3MS4xNDg2IDQwLjg1MTMgODAuMzAyOSAzMS43MjE3IDgwLjMwMjkgMjAuNTAwMUM4MC4zMDI5IDkuMjc4NDEgNzEuMTQ4NiAwLjE0ODg2NSA1OS44OTY2IDAuMTQ4ODY1Wk00MC40MTg4IDMyLjA1NTVDMzkuNzY3OCAzMi4xODk4IDM5LjAzNTIgMzIuMjE0MiAzOC41MzczIDMxLjY5NTNDMzguMzUzNiAzMi4xMTY1IDM3LjkyNTEgMzEuODk0NyAzNy41OTQ1IDMxLjgzOThDMzcuNTYzOSAzMS45MjUyIDM3LjUzNzQgMzIuMDAwNSAzNy41MDg4IDMyLjA4NkMzNi40MDg5IDMyLjE1OTMgMzUuNTg0NSAzMS4wNCAzNS4wNjAxIDMwLjE5NTRDMzQuMDE5MyAyOS42MzM3IDMyLjgzNzggMjkuMjkxOCAzMS43NzQ2IDI4LjcwMzZDMzEuNzEzNCAyOS42NzI0IDMxLjkyNTcgMzAuODczMSAzMS4wMDEyIDMxLjQ5NzlDMzAuOTU0MyAzMy4zNiAzMy44MjU1IDMxLjcxNzcgMzQuMDg4NyAzMy4xMDU2QzMzLjg4NDcgMzMuMTI4IDMzLjY1ODIgMzMuMDczIDMzLjQ5NDkgMzMuMjI5N0MzMi43NDYgMzMuOTU2MyAzMS44ODY5IDMyLjY4MDMgMzEuMDIzNyAzMy4yMDc0QzI5Ljg2NDYgMzMuNzg5NCAyOS43NDgzIDM0LjI2NTYgMjguMzEzNyAzNC4zODU3QzI4LjIzNDIgMzQuMjY1NiAyOC4yNjY4IDM0LjE4NjIgMjguMzM0MSAzNC4xMTNDMjguNzM4MiAzMy42NDQ5IDI4Ljc2NjggMzMuMDkzNCAyOS40NTY1IDMyLjg5MzlDMjguNzQ2NCAzMi43ODIgMjguMTUyNSAzMy4xNzI4IDI3LjU1NDYgMzMuNDgyMUMyNi43NzcxIDMzLjc5OTYgMjYuNzgzMyAzMi43NjU3IDI1LjU4NzUgMzMuNTM3QzI1LjQ1NDggMzMuNDI5MiAyNS41MTgxIDMzLjMzMTUgMjUuNTkzNiAzMy4yNDgxQzI1Ljg5NzYgMzIuODc3NyAyNi4yOTc2IDMyLjgyMjcgMjYuNzQ4NiAzMi44NDMxQzI0LjUzMDQgMzEuNjA5OCAyMy40ODU2IDM0LjM1MTEgMjIuNDYxMiAzMi45ODc2QzIyLjE1MzEgMzMuMDY5IDIyLjAzNjggMzMuMzQ1NyAyMS44NDI5IDMzLjU0MTFDMjEuNjc1NiAzMy4zNTggMjEuODAyMSAzMy4xMzYxIDIxLjgxMDMgMzIuOTIwNEMyMS42MTAzIDMyLjgyNjggMjEuMzU3MiAzMi43ODIgMjEuNDE2NCAzMi40NjI1QzIxLjAyNDYgMzIuMzMwMiAyMC43NTEyIDMyLjU2MjIgMjAuNDU5NCAzMi43ODJDMjAuMTk2MSAzMi41Nzg1IDIwLjYzNjkgMzIuMjgxNCAyMC43MTg1IDMyLjA2OTdDMjAuOTUzMiAzMS42NjI3IDIxLjQ4NzggMzEuOTg2MyAyMS43NTkyIDMxLjY5MzJDMjIuNTMwNiAzMS4yNTU3IDIzLjYwNiAzMS45NjU5IDI0LjQ4NzYgMzEuODQ1OUMyNS4xNjcxIDMxLjkzMTMgMjYuMDA3OCAzMS4yMzUzIDI1LjY2NyAzMC41NDEzQzI0Ljk0MDYgMjkuNjE1NCAyNS4wNjkxIDI4LjQwNDUgMjUuMDUyOCAyNy4yOTc0QzI0Ljk2MyAyNi42NTIyIDIzLjQxMDEgMjUuODMgMjIuOTYxMiAyNS4xMzRDMjIuNDA2MSAyNC41MDcyIDIxLjk3MzUgMjMuNzgwNyAyMS41NDA5IDIzLjA2NjRDMTkuOTc5OCAyMC4wNTIzIDIwLjQ3MTYgMTYuMTc5NSAxOC41MDQ0IDEzLjM4MTJDMTcuNjE0NyAxMy44NzE3IDE2LjQ1NTYgMTMuNjM5NyAxNS42ODg0IDEyLjk4MjNDMTUuMjc0MSAxMy4zNTg4IDE1LjI1NTcgMTMuODUxMyAxNS4yMjMxIDE0LjM3NDRDMTQuMjI5MyAxMy4zODMzIDE0LjM1MzggMTEuNTEwOSAxNS4xNDc2IDEwLjQwNzlDMTUuNDcyMSA5Ljk3MjM5IDE1Ljg1OTggOS42MTQyMSAxNi4yOTI0IDkuMjk4NzZDMTYuMzkwMyA5LjIyNzU0IDE2LjQyMyA5LjE1ODM0IDE2LjQyMDkgOS4wNDg0NEMxNy4yMDY2IDUuNTIzNjIgMjIuNTY1MyA2LjIwMzM1IDI0LjI1OSA4LjcwMDQ0QzI1LjQ4NzUgMTAuMjM3IDI1Ljg1ODkgMTIuMjcgMjcuMjUyNiAxMy42OTY3QzI5LjEyNzkgMTUuNzQ0IDMxLjI2NDUgMTcuNTQ3MSAzMi45OTQ5IDE5LjcyNjdDMzQuNjMxNSAyMS43MTkxIDM1LjgwMDggMjQuMDU1NCAzNi44MjExIDI2LjQxMDFDMzcuMjM3NCAyNy4xOTE1IDM3LjI0MTUgMjguMTUwMSAzNy44NTc4IDI4LjgxNzZDMzguMTYxOCAyOS4yMjA2IDM5LjY0NzQgMzAuMzE3NSAzOS4zMjUgMzAuNzA2MkMzOS41MTA3IDMxLjEwOTEgNDAuODk4MyAzMS42NjQ3IDQwLjQxNjcgMzIuMDU1NUg0MC40MTg4Wk02Ni40NDQ5IDIwLjAzMkw2My40MjY5IDIzLjA0MTlDNjIuNjIyOCAyMy44NDM4IDYxLjYyOTEgMjQuNDQyMSA2MC41NTE2IDI0Ljc2OTdMNjAuNDk2NSAyNC43ODZMNjAuNDc2MSAyNC44Mzg5QzYwLjEyNTEgMjUuNzYyOSA1OS41OTI1IDI2LjU4NzEgNTguODkwNSAyNy4yODUyTDU1Ljg3MjQgMzAuMjk1MUM1NC41NTYyIDMxLjYwNzggNTIuODA1NCAzMi4zMzAyIDUwLjk0MDIgMzIuMzMwMkM0OS4wNzUxIDMyLjMzMDIgNDcuMzI0MiAzMS42MDc4IDQ2LjAwOCAzMC4yOTUxQzQzLjI4NzkgMjcuNTgyMyA0My4yODc5IDIzLjE3MDEgNDYuMDA4IDIwLjQ1NzNMNDkuMDI2MSAxNy40NDc0QzQ5LjgzNDIgMTYuNjQxNSA1MC44MDE1IDE2LjA2MTUgNTEuODk3MyAxNS43MjU3TDUxLjk1MjQgMTUuNzA5NEw1MS45NzI4IDE1LjY1NjVDNTIuMzIzOCAxNC43MzI1IDUyLjg1ODQgMTMuOTA2MyA1My41NjI1IDEzLjIwMjFMNTYuNTgwNSAxMC4xOTIyQzU3Ljg5NjcgOC44Nzk1MyA1OS42NDc2IDguMTU3MDYgNjEuNTEyNyA4LjE1NzA2QzYzLjM3NzkgOC4xNTcwNiA2NS4xMjg3IDguODc5NTMgNjYuNDQ0OSAxMC4xOTIyQzY3Ljc2MTIgMTEuNTA0OCA2OC40ODU2IDEzLjI1MSA2OC40ODU2IDE1LjExMTFDNjguNDg1NiAxNi45NzEyIDY3Ljc2MTIgMTguNzE5MyA2Ni40NDQ5IDIwLjAzVjIwLjAzMloiIGZpbGw9IkN1cnJlbnRDb2xvciI+PC9wYXRoPgo8L3N2Zz4=
        """

        # Dictionary mapping chain types to their icons
        icons = {
            # Core LangChain types
            "LangChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "LLMChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",

            # Common LangChain chain types
            "ConversationChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "RetrievalQA": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "ConversationalRetrievalChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "SQLDatabaseChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "APIChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "TransformChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "SequentialChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "RouterChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "VectorDBQA": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "PALChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "QAWithSourcesChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "GraphQAChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "HypotheticalDocumentEmbedder": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "LLMMathChain": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
            "OpenAIFunctions": f"data:image/svg+xml;base64,{LANGCHAIN_ICON}",
        }

        # Default to LangChain icon if specific chain type not found
        return icons.get(chain_type, icons["LangChain"])

def run_example():
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
    if is_dry_run:
        print("\nRunning in DRY RUN mode - metadata will be printed but not sent to DataHub")

    # Initialize LangChain components
    llm = OpenAI(model_name='text-davinci-003')
    prompt = PromptTemplate(
        input_variables=['question'],
        template="Answer the following question:\n\n{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Initialize metadata emitter
    metadata_emitter = LangChainMetadataEmitter()

    try:
        # Emit metadata for all components
        model_urn = metadata_emitter.emit_model_metadata(llm)
        prompt_urn = metadata_emitter.emit_prompt_metadata(prompt)
        chain_urn = metadata_emitter.emit_chain_metadata(chain, model_urn, prompt_urn)

        # Run the chain (always run in both modes)
        question = "What is the capital of France?"
        answer = chain.run(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

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
