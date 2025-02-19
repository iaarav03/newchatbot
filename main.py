# main.py
import os
import io
import requests
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Using the new imports from langchain-community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.prompts.chat import ChatPromptTemplate

from PyPDF2 import PdfReader  # For PDF support

# -------------------------------------------
# Custom LLM wrapper for Groq API
# -------------------------------------------
class GroqLLM(LLM):
    api_key: str
    model: str = "llama-3.3-70b-versatile"  # default model; change as needed
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Wrap the prompt in a chat message with role "user"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        # Correct Groq API chat completions endpoint
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        # Assuming the API returns the generated text in:
        # result["choices"][0]["message"]["content"]
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

# -------------------------------------------
# FastAPI App Setup
# -------------------------------------------
app = FastAPI()

# Allow CORS for integration with your MERN frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to store session-specific chat chains
session_chat_chains = {}

# -------------------------------------------
# Upload Endpoint
# -------------------------------------------
@app.post("/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a file along with a session_id.
    Handles both text and PDF files.
    """
    # Check file extension to handle PDF files
    if file.filename.lower().endswith(".pdf"):
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    else:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "File must be a UTF-8 encoded text file."}

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    # Create Document objects
    docs = [Document(page_content=t) for t in texts]

    # Create embeddings and store in Chroma
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding)

    # Create a prompt template that uses {question} as the sole input key.
    chat_prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant that answers questions solely based on the content provided in the uploaded document.\n\n"
        "Context: {context}\n\n"
        "Conversation History: {chat_history}\n\n"
        "User Question: {question}\n\n"
        "Answer:"
    )

    # Set up conversation memory for this session
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Retrieve the Groq API key from environment variables and create the ConversationalRetrievalChain
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return {"error": "Groq API key not found in environment variables."}

    chat_chain = ConversationalRetrievalChain.from_llm(
        GroqLLM(api_key=groq_api_key, temperature=0.0),
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt}  # prompt uses {question} now
    )

    # Save the chat chain for this session (session_id ensures repeated Q&A uses the same chain)
    session_chat_chains[session_id] = chat_chain

    # Optional: Print sessions for debugging
    print("Current sessions:", session_chat_chains.keys())

    return {"message": f"File uploaded and indexed successfully for session: {session_id}"}

# -------------------------------------------
# Chat Endpoint
# -------------------------------------------
class ChatQuery(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
def chat(query: ChatQuery):
    """
    Answer a question for the provided session.
    Maintains conversation history so the user can ask multiple questions.
    """
    session_id = query.session_id
    if session_id not in session_chat_chains:
        return {"error": "No document has been uploaded for this session. Please upload a file first."}

    chat_chain = session_chat_chains[session_id]
    # Now pass only the "question" key, as expected by the prompt template and memory.
    result = chat_chain({"question": query.question})
    return result

# -------------------------------------------
# Running the App
# -------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
