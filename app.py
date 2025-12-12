import streamlit as st
import os
from dotenv import load_dotenv

# Load our backend modules
from src.ingestion import load_pdf_documents
from src.processing import summarize_data
from src.retrieval import build_vector_store
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Qatar IMF Analyst", layout="wide")
st.title("🤖 Qatar IMF Report: Multi-Modal RAG Agent")

# -- 1. Sidebar: The Setup Button --
with st.sidebar:
    st.header("Settings")
    if st.button("Process Document (Ingest)"):
        with st.spinner("Parsing PDF (This takes 2-3 mins)..."):
            # A. Ingestion
            texts, tables = load_pdf_documents("data/qatar_test_doc.pdf")
            st.write(f"Found {len(texts)} text chunks and {len(tables)} tables.")
            
            # B. Processing (Summarization)
            table_summaries = []
            progress_bar = st.progress(0)
            for i, table in enumerate(tables):
                # Update progress bar
                progress_bar.progress((i + 1) / len(tables))
                summary = summarize_data(table, "table")
                table_summaries.append(summary)
            
            # C. Indexing
            st.session_state.retriever = build_vector_store(texts, tables, table_summaries)
            st.success("System Ready!")

# -- 2. Main Chat Interface --
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Qatar's fiscal policy..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            # 1. Retrieve context
            docs = st.session_state.retriever.invoke(prompt)
            context_text = "\n\n".join([str(d) for d in docs])
            
            # 2. Generate Answer
            template = """Answer the question based ONLY on the following context. 
            If the context contains tables, use the data in the tables.
            
            Context:
            {context}
            
            Question: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))
            chain = prompt_template | model | StrOutputParser()
            
            response = chain.invoke({"context": context_text, "question": prompt})
            
            st.markdown(response)
            
            # Show sources (Bonus points for transparency)
            with st.expander("View Source Documents"):
                st.markdown(context_text)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please process the document first!")