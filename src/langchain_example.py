import os
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent

load_dotenv()

class LangChainMetadataEmitter:
    def __init__(self, gms_server="http://localhost:8080"):
        self.emitter = DatahubRestEmitter(gms_server)

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
        self.emitter.emit(mce)
        return model_urn

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
        self.emitter.emit(mce)
        return prompt_urn

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
        self.emitter.emit(mce)
        return chain_urn

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
        PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMiA3TDEyIDEyTDIyIDdMMTIgMloiIGZpbGw9IiMwMEIzRTYiLz4KPHBhdGggZD0iTTIgMTdMMTIgMjJMMjIgMTdMMTIgMTJMMiAxN1oiIGZpbGw9IiMwMEIzRTYiLz4KPC9zdmc+Cg==
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
    # Initialize LangChain components
    llm = OpenAI(model_name='text-davinci-003')
    prompt = PromptTemplate(
        input_variables=['question'],
        template="Answer the following question:\n\n{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Initialize metadata emitter
    metadata_emitter = LangChainMetadataEmitter()

    # Emit metadata for all components
    model_urn = metadata_emitter.emit_model_metadata(llm)
    prompt_urn = metadata_emitter.emit_prompt_metadata(prompt)
    chain_urn = metadata_emitter.emit_chain_metadata(chain, model_urn, prompt_urn)

    # Run the chain
    question = "What is the capital of France?"
    answer = chain.run(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Metadata URNs:")
    print(f"Model: {model_urn}")
    print(f"Prompt: {prompt_urn}")
    print(f"Chain: {chain_urn}")

if __name__ == "__main__":
    run_example()
