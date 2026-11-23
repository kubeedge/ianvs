import os
from typing import List, Optional, Union
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from tqdm import tqdm


def default_device() -> str:
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GovernmentRAG:
    def __init__(
        self,
        base_path: str = "./dataset/gov_rag",
        provinces: Optional[Union[str, List[str]]] = None,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
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
        if not os.path.isdir(os.path.join(self.base_path, "dataset")):
            raise FileNotFoundError(
                f"Knowledge base not found at {os.path.join(self.base_path, 'dataset')}. "
                "Download the GovAff dataset from "
                "https://www.kaggle.com/datasets/kubeedgeianvs/the-government-affairs-dataset-govaff "
                "and place it as described in examples/government_rag/README.md"
            )
        self.provinces = self._validate_provinces(provinces)
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device or default_device()}
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
    
    def _load_documents(self, province_path: str, province: str) -> List:
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

        # Tag each document so retrieval can be restricted by province even
        # when the full persisted store is reloaded later.
        for doc in documents:
            doc.metadata["province"] = province

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
                documents = self._load_documents(province_path, province)
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

        # Restrict retrieval to the selected provinces. The persisted store
        # may contain documents from all provinces, so relying on what was
        # ingested is not enough.
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"province": {"$in": self.provinces}},
            }
        )
        
        # Get relevant documents
        docs = retriever.invoke(query)
        
        # Format the response
        response = "Relevant information:\n\n"
        for i, doc in enumerate(docs, 1):
            response += f"Document {i}:\n{doc.page_content}\n\n"
            
        return response 