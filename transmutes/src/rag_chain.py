"""RAG chain for question answering with Eastern wisdom."""

from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .prompts import (
    get_rag_prompt,
    get_conversational_rag_prompt,
    get_condense_question_prompt
)


class RAGChain:
    """Retrieval-Augmented Generation chain for Eastern philosophy."""
    
    def __init__(
        self,
        retriever,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        use_conversation_memory: bool = True
    ):
        """Initialize RAG chain.
        
        Args:
            retriever: Document retriever from vector store.
            model_name: Name of the HuggingFace model to use.
            temperature: Sampling temperature for generation.
            max_new_tokens: Maximum number of tokens to generate.
            use_conversation_memory: Whether to use conversation memory.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_conversation_memory = use_conversation_memory
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize chain
        if use_conversation_memory:
            self.chain = self._create_conversational_chain()
        else:
            self.chain = self._create_simple_chain()
    
    def _initialize_llm(self) -> LLM:
        """Initialize the language model.
        
        Returns:
            LangChain LLM object.
        """
        print(f"Loading language model: {self.model_name}")
        print("Note: This may take a while and requires significant memory.")
        print("For faster inference, consider using a smaller model or API-based LLM.\n")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            print("Language model loaded successfully.\n")
            return llm
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nFalling back to a simpler approach...")
            print("Consider using an API-based model (OpenAI, Anthropic, etc.) for better results.")
            raise
    
    def _create_simple_chain(self) -> RetrievalQA:
        """Create a simple RAG chain without conversation memory.
        
        Returns:
            RetrievalQA chain.
        """
        prompt = get_rag_prompt()
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
    
    def _create_conversational_chain(self) -> ConversationalRetrievalChain:
        """Create a conversational RAG chain with memory.
        
        Returns:
            ConversationalRetrievalChain.
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
            condense_question_prompt=get_condense_question_prompt(),
            combine_docs_chain_kwargs={
                "prompt": get_conversational_rag_prompt()
            }
        )
        
        return chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG chain.
        
        Args:
            question: User's question.
            
        Returns:
            Dictionary with answer and source documents.
        """
        if self.use_conversation_memory:
            result = self.chain({"question": question})
        else:
            result = self.chain({"query": question})
        
        return result
    
    def get_answer(self, question: str) -> str:
        """Get just the answer text.
        
        Args:
            question: User's question.
            
        Returns:
            Answer text.
        """
        result = self.query(question)
        return result.get('answer', result.get('result', ''))
    
    def get_answer_with_sources(self, question: str) -> tuple[str, List[str]]:
        """Get answer with source documents.
        
        Args:
            question: User's question.
            
        Returns:
            Tuple of (answer, list of source titles).
        """
        result = self.query(question)
        answer = result.get('answer', result.get('result', ''))
        
        sources = []
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source = doc.metadata.get('title', 'Unknown Source')
                if source not in sources:
                    sources.append(source)
        
        return answer, sources
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.use_conversation_memory and hasattr(self.chain, 'memory'):
            self.chain.memory.clear()




