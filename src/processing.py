import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CRITICAL FIX: Load env vars IMMEDIATELY ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Debug check: Print to terminal if key is found
if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
else:
    print("✅ API Key loaded successfully.")
# -----------------------------------------------

# Pass the key directly to the model to avoid any confusion
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0,
    google_api_key=api_key  # <--- THIS IS THE MISSING PIECE
)

def summarize_data(element, data_type="table"):
    """
    Takes raw data (text or table HTML) and asks the LLM to write a summary.
    """
    if data_type == "text":
        return str(element) 

    prompt_text = """
    You are a financial analyst. 
    Explain the following table in clear sentences. 
    Highlight the key numbers, trends, and column headers.
    Do not lose important data points.
    
    Table Data: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | model | StrOutputParser()
    
    summary = chain.invoke({"element": str(element)})
    return summary