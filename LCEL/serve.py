from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set. Please set it in your .env file.")

# Check if model name is valid (replace with a valid model if needed)
MODEL_NAME = "Gemma2-9b-It"  # Change to a valid model if this fails
try:
    model = ChatGroq(model=MODEL_NAME, groq_api_key=groq_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatGroq model '{MODEL_NAME}': {e}")

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser=StrOutputParser()

##create chain
chain=prompt_template|model|parser



## App definition
app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

# Health check endpoint for debugging
@app.get("/health")
def health():
    return {"status": "ok"}

## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)


