# 🤖 Multi-Modal RAG Agent: IMF Financial Analyst

This project is a Retrieval-Augmented Generation (RAG) system built to analyze complex financial documents. It was developed as part of an AI/ML Internship assessment to demonstrate how to handle multi-modal data (text + tables) effectively.

## 🎯 The Problem
Standard RAG systems struggle with PDFs containing financial tables. They often chop tables into meaningless chunks, causing the AI to hallucinate numbers or lose context.

## 💡 My Solution
I built a **Multi-Vector Retrieval Architecture** that decouples the "Search" from the "Generation":
1.  **Ingestion:** I used `unstructured` to separate text from tables.
2.  **Processing:** I used **Google Gemini 2.5 Flash** to write natural language summaries for every table.
3.  **Retrieval:** The system searches the *summary* (to find the right table) but feeds the *original raw table* to the LLM (to ensure math accuracy).

## 🛠️ Tech Stack
* **LLM:** Google Gemini 2.5 Flash (via `langchain-google-genai`)
* **Orchestration:** LangChain
* **Database:** ChromaDB (Vector Store) + InMemoryStore (Doc Store)
* **Frontend:** Streamlit

## 🚀 How to Run Locally

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/qatar-imf-rag-agent.git](https://github.com/YOUR_USERNAME/qatar-imf-rag-agent.git)
    cd qatar-imf-rag-agent
    ```

2.  **Install Dependencies**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Set up API Key**
    Create a `.env` file and add your Google Key:
    ```text
    GOOGLE_API_KEY=your_key_here
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📊 Sample Query
**Question:** "What is the projected fiscal surplus for 2024?"
**Answer:** "0.3% of GDP" (Correctly retrieved from Table 1).
