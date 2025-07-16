from langchain_chroma import Chroma
from utils.utility import Helperclass
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class VectorStoreManager:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = None

    def load_vectorstore(self, embedding_function):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"Loaded vector store from '{self.persist_directory}'")
        return self.vectorstore

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

    def run(self, question: str):
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.azure_llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(question)
        print(f"Response from RAG chain: {response}")
        return response

class RAGPipeline:
    def __init__(self, collection_name: str, embedding_function, persist_directory: str = "./chroma_db"):
        self.vectorstore_manager = VectorStoreManager(collection_name, persist_directory)
        self.embedding_function = embedding_function

    def load_vectorstore(self):
        self.vectorstore_manager.load_vectorstore(self.embedding_function)

    def query(self, query: str, k: int = 10):
        retriever = self.vectorstore_manager.get_retriever(k)
        rag_chain = RAGChain(retriever)
        return rag_chain.run(query)

if __name__ == "__main__":
    collection_name = "pradeep_rag"
    persist_directory = "./chroma_db"
    query = "Give me 2 names who are well experianced in Pyspark. Also give me references from where this was taken"

    # Get embedding function from your helper (Gemini, Azure, etc.)
    helper = Helperclass()
    embedding_function = helper.gemini_client()  # or helper.openai_client() for Azure

    rag_pipeline = RAGPipeline(collection_name, embedding_function, persist_directory)
    rag_pipeline.load_vectorstore()
    rag_pipeline.query(query, k=20)