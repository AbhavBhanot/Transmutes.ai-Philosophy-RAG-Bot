"""Document ingestion script for building the vector store."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import ConfigLoader
from src.document_loader import PhilosophyDocumentLoader
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStoreManager


def ingest_documents(config_path: str = "config.yaml", force_rebuild: bool = False):
    """Ingest documents and build vector store.
    
    Args:
        config_path: Path to configuration file.
        force_rebuild: If True, rebuild vector store even if it exists.
    """
    print("=" * 60)
    print("Transmutes AI - Document Ingestion")
    print("=" * 60)
    print()
    
    # Load configuration
    config = ConfigLoader(config_path)
    
    # Initialize document loader
    data_dir = config.data_directory
    doc_loader = PhilosophyDocumentLoader(str(data_dir))
    
    # Load all documents
    print(f"Loading documents from: {data_dir}")
    documents = doc_loader.load_all_documents()
    
    if not documents:
        print("No documents found. Please check your data directory.")
        return
    
    # Show statistics
    stats = doc_loader.get_document_statistics(documents)
    print("\nDocument Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Unique sources: {stats['unique_sources']}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Average length: {stats['avg_document_length']:,} characters")
    print(f"  Document types: {stats['document_types']}")
    print()
    
    # Initialize embeddings manager
    embedding_model = config.get('embeddings.model_name')
    chunk_size = config.get('text_splitter.chunk_size', 800)
    chunk_overlap = config.get('text_splitter.chunk_overlap', 200)
    
    embeddings_manager = EmbeddingsManager(
        model_name=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents into chunks
    split_docs = embeddings_manager.split_documents(documents)
    print()
    
    # Initialize vector store manager
    persist_dir = config.persist_directory
    collection_name = config.get('vector_store.collection_name')
    
    vector_store_manager = VectorStoreManager(
        embeddings=embeddings_manager.get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=collection_name
    )
    
    # Create or load vector store
    if force_rebuild:
        print("Force rebuild enabled. Creating new vector store...")
        # Delete existing store if present
        if persist_dir.exists():
            import shutil
            shutil.rmtree(persist_dir)
        vector_store_manager.create_vector_store(split_docs)
    else:
        vector_store_manager.get_or_create_vector_store(split_docs)
    
    print()
    print("=" * 60)
    print("Ingestion Complete!")
    print(f"Vector store saved to: {persist_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents for Transmutes AI")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild vector store"
    )
    
    args = parser.parse_args()
    
    ingest_documents(config_path=args.config, force_rebuild=args.rebuild)




