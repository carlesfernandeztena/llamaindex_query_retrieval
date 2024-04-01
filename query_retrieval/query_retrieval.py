"""Routines for query retrieval using LlamaIndex"""

import os
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.llms.gemini import Gemini

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def store_documents_from_folder(
    docstore: SimpleDocumentStore, folder: str = "./data"
) -> SimpleDocumentStore:
    """Collect documents from folder and store them"""
    # Read all docs from a folder
    documents = SimpleDirectoryReader(folder).load_data()
    # Split docs into nodes
    nodes = SentenceSplitter().get_nodes_from_documents(documents)
    # Store nodes in given document store, and return it
    docstore.add_documents(nodes)
    return docstore, nodes


def get_local_embed_model():
    """Set a local model for embeddings"""
    local_model_path = Path("./bge_onnx")
    if not local_model_path.exists():
        # Download embedding model from HF and save locally
        OptimumEmbedding.create_and_save_optimum_model(
            "BAAI/bge-small-en-v1.5", str(local_model_path)
        )
    # Set this model for embedding extraction
    return OptimumEmbedding(folder_name=str(local_model_path))


def run_query_retrieval(
    query: str,
    indexing_mode: str,
    data_folder: str,
    temperature: float,
    chunk_size: int,
    similarity_top_k: int,
) -> None:
    """Run query retrieval on document folder using LlamaIndex.

    Args:
        query (str): Query to be used for retrieval.
        indexing_mode (str): Choose the indexing mode: summary, vector, or keyword table.
        data_folder (str): Directory containing the documents to be parsed.
        temperature (float): Temperature value for the LLM.
        chunk_size (int): Set the chunk size for processing.
        similarity_top_k (int): Set the top K similarities to consider.
    """

    # Set initial settings
    print(":: Initializing settings")
    Settings.llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model_name="models/gemini-pro",
        temperature=temperature,
    )
    Settings.chunk_size = chunk_size

    # Parse input documents, prepare storage context
    print(":: Processing folder documents and storing embeddings")
    docstore = SimpleDocumentStore()
    docstore, nodes = store_documents_from_folder(docstore, data_folder)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    print(f":: Number of stored nodes: {len(storage_context.docstore.docs)}")

    # Set a local embedding model (to save credits)
    Settings.embed_model = get_local_embed_model()

    if indexing_mode == "summary":
        index = SummaryIndex(nodes, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query)
        print("*" * 50)
        print(f"Summary index response:\n{response}")

    elif indexing_mode == "vector":
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print("*" * 50)
        print(f"Vector index response:\n{response}")

    elif indexing_mode == "keyword":
        index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print("*" * 50)
        print(f"Keyword table response:\n{response}")

    else:
        print("Invalid indexing mode. Choose from 'summary', 'vector', or 'keyword'.")
