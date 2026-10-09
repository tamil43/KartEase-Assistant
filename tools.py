import os
import csv
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

# --- TOOL 1: Order Status Lookup ---
def get_order_status(order_id: str) -> str:
    """Look up the product, status, date, and payment method for a given KartEase order ID (e.g. KE1001)."""
    file_path = "orders.csv"
    if not os.path.exists(file_path):
        return "Orders database not found."
    
    order_id_clean = order_id.strip().upper()
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("order_id", "").strip().upper() == order_id_clean:
                return (
                    f"Order ID: {row['order_id']}, "
                    f"Product: {row['product']}, "
                    f"Status: {row['status']}, "
                    f"Date: {row['date']}, "
                    f"Payment Method: {row['payment_method']}"
                )
    return f"No order found with ID {order_id}."

# --- TOOL 2: Policy Search ---
def search_policies(question: str) -> str:
    """Search KartEase return, shipping, warranty, and payment policies to answer customer questions. Always name the source file."""
    persist_dir = "./chroma_db"
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_collection("kartease_policies")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(question)
    
    if not nodes:
        return "Sorry, I can only help with KartEase orders and policies."
    
    context_blocks = []
    for node in nodes:
        # Extract file name from metadata safely
        file_path = node.metadata.get("file_path", node.metadata.get("file_name", "policy document"))
        file_name = os.path.basename(file_path)
        context_blocks.append(f"\n{node.text}")
        
    return "\n\n".join(context_blocks)

# Wrap into LlamaIndex FunctionTools
order_tool = FunctionTool.from_defaults(fn=get_order_status)
policy_tool = FunctionTool.from_defaults(fn=search_policies)