import uuid
import os
from dotenv import load_dotenv

# --- Load env vars ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# ---------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore # <--- CHANGED THIS

# Robust Import Block for Retriever
try:
    from langchain.retrievers import MultiVectorRetriever
except ImportError:
    from langchain.retrievers.multi_vector import MultiVectorRetriever

def build_vector_store(text_elements, table_elements, table_summaries):
    """
    Creates the database. Indexes the summaries, but returns the raw data.
    """
    # 1. The Vector Database (Search Index)
    vectorstore = Chroma(
        collection_name="qatar_report",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        ),
        persist_directory="./vector_store"
    )
    
    # 2. The Document Store (Storage Layer)
    # We use InMemoryStore because it handles objects better than ByteStore
    store = InMemoryStore()
    
    id_key = "doc_id"
    
    # -- Handle Texts --
    doc_ids = [str(uuid.uuid4()) for _ in text_elements]
    
    # Vectorstore gets the summary (or text itself)
    summary_texts = [
        Document(page_content=str(t), metadata={id_key: doc_ids[i]}) 
        for i, t in enumerate(text_elements)
    ]
    vectorstore.add_documents(summary_texts)
    
    # Docstore gets the ORIGINAL content wrapped as a Document
    # CRITICAL FIX: Convert 'CompositeElement' to string using str(t)
    real_text_docs = [
        Document(page_content=str(t), metadata={"type": "text"})
        for t in text_elements
    ]
    store.mset(list(zip(doc_ids, real_text_docs)))

    # -- Handle Tables --
    table_ids = [str(uuid.uuid4()) for _ in table_elements]
    
    # Vectorstore gets the SUMMARY
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]}) 
        for i, s in enumerate(table_summaries)
    ]
    vectorstore.add_documents(summary_tables)
    
    # Docstore gets the RAW TABLE (as a string)
    # CRITICAL FIX: Convert 'Table' object to string using str(t)
    real_table_docs = [
        Document(page_content=str(t), metadata={"type": "table"})
        for t in table_elements
    ]
    store.mset(list(zip(table_ids, real_table_docs)))

    # 3. Create the Retriever Link
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store, # Note: param name is 'docstore' or 'byte_store' depending on version, usually docstore works for both
        id_key=id_key,
    )
    
    return retriever