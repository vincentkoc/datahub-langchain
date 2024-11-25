from typing import List, Dict
import os
from datetime import datetime
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

from src.config import ObservabilityConfig, ObservabilitySetup
from src.utils.metrics import MetricsAggregator
from src.utils.lineage import LineageTracker

def load_documents(file_path: str) -> List[str]:
    """Load and split documents"""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)

def setup_rag_chain(docs: List[str], observer):
    """Setup RAG chain with observation"""
    # Initialize components
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(
        temperature=0,
        callbacks=[observer]
    )

    # Create and return chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        callbacks=[observer]
    )

    return chain

def main():
    # Load configuration
    config = ObservabilityConfig.from_env()

    # Initialize observability
    obs = ObservabilitySetup(config)
    obs.setup()

    # Get LangChain observer
    observer = obs.get_observer("langchain")

    # Setup metrics and lineage tracking
    metrics = MetricsAggregator()
    lineage = LineageTracker()

    # Load documents from configured path
    data_path = Path(os.getenv("DATA_PATH", "data/sample.txt"))
    docs = load_documents(str(data_path))

    # Setup and run chain
    chain = setup_rag_chain(docs, observer)

    # Run query
    query = "What is the capital of France?"
    start_time = datetime.now()
    result = chain({"query": query})
    end_time = datetime.now()

    # Log metrics
    metrics.add_metrics({
        "latency": (end_time - start_time).total_seconds(),
        "n_source_docs": len(result.get("source_documents", [])),
        "response_length": len(result.get("result", ""))
    })

    # Print results
    print("\nQuery:", query)
    print("Answer:", result["result"])
    print("\nSource Documents:")
    for doc in result.get("source_documents", []):
        print(f"- {doc.page_content[:100]}...")

    # Print metrics
    print("\nMetrics:")
    print(metrics.get_stats("latency"))

if __name__ == "__main__":
    main()
