import os
from typing import List, Optional, Union
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

class GovernmentRAG:
    def __init__(
        self,
        base_path: str = "/path/ianvs/dataset/gov_rag",
        provinces: Optional[Union[str, List[str]]] = None,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the Government RAG system.
        
        Args:
            base_path: Base path to the government documents
            provinces: List of provinces to include, or 'all' for all provinces
            model_name: Name of the embedding model to use
            device: Device to run the model on
            persist_directory: Directory to persist the vector database
        """
        self.base_path = base_path
        self.provinces = self._validate_provinces(provinces)
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        self.vector_store = None
        self._initialize_knowledge_base()
        
    def _validate_provinces(self, provinces: Optional[Union[str, List[str]]]) -> List[str]:
        """Validate and process the provinces parameter."""
        if provinces == "all" or provinces is None:
            # Get all province directories
            return [d for d in os.listdir(os.path.join(self.base_path, "dataset")) 
                   if os.path.isdir(os.path.join(self.base_path, "dataset", d))]
        elif isinstance(provinces, str):
            return [provinces]
        elif isinstance(provinces, list):
            return provinces
        else:
            raise ValueError("provinces must be 'all', a string, or a list of strings")
    
    def _load_documents(self, province_path: str) -> List:
        """Load documents from a specific province directory."""
        loaders = []
        
        # Load .txt files
        txt_loader = DirectoryLoader(
            province_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        loaders.append(txt_loader)
        
        # Load .docx files
        docx_loader = DirectoryLoader(
            province_path,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader
        )
        loaders.append(docx_loader)
        
        # Load all documents
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading documents from {province_path}: {str(e)}")
                
        return documents
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base by loading and processing documents."""

        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector database from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return

        all_documents = []
        
        # Load documents from each selected province with progress bar
        print("Loading documents from provinces...")
        for province in tqdm(self.provinces, desc="Processing provinces"):
            province_path = os.path.join(self.base_path, "dataset", province)
            if os.path.exists(province_path):
                documents = self._load_documents(province_path)
                all_documents.extend(documents)
        
        if not all_documents:
            raise ValueError("No documents found in the specified provinces")
        
        # Split documents into chunks with progress bar
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_documents)
        
        # Create vector store with persistence
        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        self.vector_store.persist()
        print(f"Vector database saved to {self.persist_directory}")
    
    def query(self, query: str, k: int = 4) -> str:
        """
        Query the knowledge base.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            Retrieved information
        """
        if not self.vector_store:
            raise ValueError("Knowledge base not initialized")
            
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Format the response
        response = "Relevant information:\n\n"
        for i, doc in enumerate(docs, 1):
            response += f"Document {i}:\n{doc.page_content}\n\n"
            
        return response 