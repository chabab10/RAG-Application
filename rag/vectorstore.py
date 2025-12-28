from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"

def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Toujours ajouter les nouveaux documents
    vectordb.add_documents(chunks)

    return vectordb
