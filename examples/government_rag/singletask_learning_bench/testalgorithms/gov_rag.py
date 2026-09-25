# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Government RAG module for retrieving relevant documents."""

import os
from typing import List, Optional, Union

# pylint: disable=import-error
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm


# pylint: disable=too-few-public-methods
class GovernmentRAG:
    """Retrieval-Augmented Generation helper for government policy documents."""

    def __init__(
        self,
        base_path: str = "./dataset/gov_rag",
        provinces: Optional[Union[str, List[str]]] = None,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        persist_directory: str = "./chroma_db"
    ):
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
            dataset_dir = os.path.join(self.base_path, "dataset")
            if not os.path.exists(dataset_dir):
                return []
            return [d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d))]
        if isinstance(provinces, str):
            return [provinces]
        if isinstance(provinces, list):
            return provinces
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
            # pylint: disable=broad-exception-caught
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
            print("No documents found in the specified provinces.")
            return

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

        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        print(f"Vector database saved to {self.persist_directory}")

    def query(self, query: str, k: int = 4) -> str:
        """Query the knowledge base."""
        if not self.vector_store:
            raise ValueError("Knowledge base not initialized")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        # Get relevant documents
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)

        # Format the response
        response = "Relevant information:\n\n"
        for i, doc in enumerate(docs, 1):
            response += f"Document {i}:\n{doc.page_content}\n\n"

        return response

# end of file