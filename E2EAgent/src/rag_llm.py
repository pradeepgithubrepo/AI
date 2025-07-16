from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.utility import Helperclass


class DocumentLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_documents(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                print(f"Unsupported file type: {filename}")
                continue
            documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents from the folder.")
        return documents

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split(self, documents: List[Document]) -> List[Document]:
        splits = self.splitter.split_documents(documents)
        print(f"Split the documents into {len(splits)} chunks.")
        if splits:
            print("First chunk source:", splits[0].metadata['source'])
        return splits

class EmbeddingManager:
    def __init__(self, helper: Helperclass):
        self.embeddings = helper.gemini_client()

    def embed_documents(self, splits: List[Document]):
        document_embeddings = self.embeddings.embed_documents([split.page_content for split in splits])
        print(f"Created embeddings for {len(document_embeddings)} document chunks.")
        return self.embeddings

class VectorStoreManager:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create_vectorstore(self, splits: List[Document], embedding_function):
        self.vectorstore = Chroma.from_documents(
            collection_name=self.collection_name,
            documents=splits,
            embedding=embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"Vector store created and persisted to '{self.persist_directory}'")
        return self.vectorstore

    def load_vectorstore(self, embedding_function):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"Loaded vector store from '{self.persist_directory}'")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 10):
        search_results = self.vectorstore.similarity_search(query, k=k)
        print(f"\nTop {k} most relevant chunks for the query: '{query}'\n")
        for i, result in enumerate(search_results, 1):
            print(f"Result {i}:")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            print(f"Content: {result.page_content}")
        return search_results

    def get_retriever(self, k: int = 10):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        print("\nRetriever object created for similarity search.")
        return retriever

class RAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """
        )
        self.azure_llm = Helperclass().openai_client()
    
    def docs2str(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, question: str):
        rag_chain = (
            {"context": self.retriever , "question": RunnablePassthrough()}
            | self.prompt
            | self.azure_llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(question)
        print(f"Response from RAG chain: {response}")
        return response

class RAGPipeline:
    def __init__(self, folder_path: str, collection_name: str):
        self.helper = Helperclass()
        self.loader = DocumentLoader(folder_path)
        self.splitter = TextSplitter()
        self.embedding_manager = EmbeddingManager(self.helper)
        self.vectorstore_manager = VectorStoreManager(collection_name)

    def build_vectorstore(self):
        documents = self.loader.load_documents()
        splits = self.splitter.split(documents)
        embedding_function = self.embedding_manager.embed_documents(splits)
        self.vectorstore_manager.create_vectorstore(splits, embedding_function)
        return embedding_function

    def load_vectorstore(self, embedding_function):
        self.vectorstore_manager.load_vectorstore(embedding_function)

    def query(self, query: str, k: int = 10):
        retriever = self.vectorstore_manager.get_retriever(k)
        rag_chain = RAGChain(retriever)
        return rag_chain.run(query)

if __name__ == "__main__":
    folder_path = "./profiles/"
    collection_name = "pradeep_rag"
    query = "Give me 2 names who are well experianced in Pyspark. Also give me references from where this was taken"

    rag_pipeline = RAGPipeline(folder_path, collection_name)
    embedding_function = rag_pipeline.build_vectorstore()
    rag_pipeline.load_vectorstore(embedding_function)
    rag_pipeline.query(query, k=20)