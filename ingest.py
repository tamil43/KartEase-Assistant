import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

def build_knowledge_base():
    persist_dir = "./chroma_db"
    db = chromadb.PersistentClient(path=persist_dir)
    
    collection_name = "kartease_policies"
    existing_collections = [c.name for c in db.list_collections()]
    
    if collection_name in existing_collections:
        print("Vector store already exists! Skipping re-embedding to save API quota.")
    else:
        print("Building vector store from policy documents...")
        chroma_collection = db.create_collection(collection_name)
        
        # Load the 4 policy documents from the data/ folder
        documents = SimpleDirectoryReader("data").load_data()
        print(f"Loaded {len(documents)} document pages/files.")
        
        # Split documents into chunks
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Split documents into {len(nodes)} chunks.")
        
        # Set up Gemini embeddings and store in Chroma
        embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        _ = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        print("Ingestion complete! Vectors saved to", persist_dir)

if __name__ == "__main__":
    build_knowledge_base()