from fastapi import FastAPI
from pydantic import BaseModel
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Initialize the NEW Gemini client
# It automatically detects and uses the GEMINI_API_KEY from your .env file
client = genai.Client()

# 3. Initialize the FastAPI application
app = FastAPI(title="Career AI Assistant Backend")

# 4. Define the data structure for incoming user messages
class ChatRequest(BaseModel):
    message: str

# 5. The Core Persona (System Prompt)
# This dictates how the AI behaves and what it knows about you
SYSTEM_PROMPT = """You are the official Career Assistant AI for Alperen Ulukaya.
Your job is to communicate with potential employers on his behalf.
You must maintain a highly professional, concise, and polite tone at all times.

Here is Alperen's professional background:
- 3rd-year Computer Engineering student at Akdeniz University in Antalya.
- Full-Stack Developer and AI Enthusiast.
- Core Tech Stack: Python, Spring Boot, Angular, .NET.
- Experience: Developed the backend architecture for an AI-based application during his internship at Talya Bilişim.
- Current Engagements: Participant in the prestigious Defense Industry 401 Education Program.
- Volunteering: Computer educator for children at ANTÇEV.

Interaction Rules:
- Answer interview invitations enthusiastically and professionally.
- Respond to technical questions accurately using his core tech stack.
- If asked about a topic outside this scope (e.g., specific salary expectations, deep legal questions, or tools he doesn't know), politely state that you do not have that specific information and will note it down for Alperen.
"""

# 6. Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Backend is running flawlessly with the NEW google-genai SDK!"}

# 7. The main chat endpoint
@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    # We call the new Gemini 2.5 Flash model and inject the System Prompt
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=request.message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3, # Low temperature keeps the AI professional and focused
        )
    )
    
    return {"response": response.text}