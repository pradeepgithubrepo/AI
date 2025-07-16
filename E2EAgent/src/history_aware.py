from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from langchain_chroma import Chroma
from utils.utility import Helperclass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

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
        return self.vectorstore

    def get_retriever(self, k: int = 10):
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

class RAGChain:
    def __init__(self, retriever):
        self.azure_llm = Helperclass().openai_client()
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        self.history_aware_retriever = create_history_aware_retriever(
            self.azure_llm,
            retriever,
            contextualize_q_prompt
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.question_answer_chain = create_stuff_documents_chain(self.azure_llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def run(self, queries):
        chat_history = []
        for question in queries:
            answer = self.rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
            print(f"Human: {question}")
            print(f"AI: {answer}\n")
            chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
        return chat_history

class RAGPipeline:
    def __init__(self, collection_name: str, embedding_function, persist_directory: str = "./chroma_db"):
        self.vectorstore_manager = VectorStoreManager(collection_name, persist_directory)
        self.embedding_function = embedding_function

    def load_vectorstore(self):
        self.vectorstore_manager.load_vectorstore(self.embedding_function)

    def query(self, queries, k: int = 10):
        retriever = self.vectorstore_manager.get_retriever(k)
        rag_chain = RAGChain(retriever)
        return rag_chain.run(queries)

if __name__ == "__main__":
    collection_name = "pradeep_rag"
    persist_directory = "./chroma_db"
    helper = Helperclass()
    embedding_function = helper.gemini_client()  # or helper.openai_client() for Azure

    queries = [
        "Who are the guys specialized in pyspark and databricks?",
        "For the above guys what other skills do they have, tabulate it?",
        "Give me 2 names who are well experienced in Pyspark. Also give me references from where this was taken"
    ]

    rag_pipeline = RAGPipeline(collection_name, embedding_function, persist_directory)
    rag_pipeline.load_vectorstore()
    rag_pipeline.query(queries, k=20)