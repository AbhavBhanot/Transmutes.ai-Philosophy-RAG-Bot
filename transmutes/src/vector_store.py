"""Vector store management for document retrieval."""

from typing import List, Optional
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


class VectorStoreManager:
    """Manage ChromaDB vector store for document retrieval."""
    
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "eastern_philosophy"
    ):
        """Initialize vector store manager.
        
        Args:
            embeddings: Embeddings object for vectorization.
            persist_directory: Directory to persist the vector store.
            collection_name: Name of the collection.
        """
        self.embeddings = embeddings
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.vector_store: Optional[Chroma] = None
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents.
        
        Args:
            documents: List of documents to index.
            
        Returns:
            Chroma vector store.
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vector_store
    
    def load_vector_store(self) -> Chroma:
        """Load existing vector store from disk.
        
        Returns:
            Chroma vector store.
        """
        print(f"Loading vector store from {self.persist_directory}...")
        
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        print("Vector store loaded successfully.")
        return self.vector_store
    
    def get_or_create_vector_store(self, documents: Optional[List[Document]] = None) -> Chroma:
        """Get existing vector store or create a new one.
        
        Args:
            documents: Documents to use if creating new store.
            
        Returns:
            Chroma vector store.
        """
        # Check if vector store exists
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        
        if chroma_db_path.exists():
            print("Existing vector store found.")
            return self.load_vector_store()
        else:
            if documents is None:
                raise ValueError("Documents must be provided to create a new vector store.")
            print("No existing vector store found. Creating new one...")
            return self.create_vector_store(documents)
    
    def add_documents(self, documents: List[Document]):
        """Add documents to existing vector store.
        
        Args:
            documents: Documents to add.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully.")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query.
            k: Number of results to return.
            
        Returns:
            List of similar documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """Get a retriever object for the vector store.
        
        Args:
            search_type: Type of search ('similarity' or 'mmr').
            k: Number of documents to retrieve.
            
        Returns:
            Retriever object.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )




