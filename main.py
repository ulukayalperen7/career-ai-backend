from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import agents
import tools
import json
import uuid
import time
from typing import List, Dict, Optional

app = FastAPI(title="Career Assistant Agentic System")

# CORS settings (Allow frontend to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open to all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory Implementation: Session-based storage
# Structure: {session_id: {"history": [], "last_seen": timestamp}}
sessions: Dict[str, Dict] = {}
MAX_HISTORY = 10
SESSION_TIMEOUT = 3600  # 1 hour

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

def cleanup_sessions():
    """Remove old sessions to prevent memory leaks."""
    current_time = time.time()
    expired = [sid for sid, data in sessions.items() if current_time - data["last_seen"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

@app.post("/chat", response_model=ChatResponse)
async def run_agent_system(request: ChatRequest):
    # cleanup old sessions occasionally
    if len(sessions) > 100:
        cleanup_sessions()

    user_query = request.message
    session_id = request.session_id

    # Create new session if needed
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"history": [], "last_seen": time.time()}
    
    # Update last seen
    sessions[session_id]["last_seen"] = time.time()
    
    # Get history
    history = sessions[session_id]["history"]

    # Only notify on critical keywords or very first message of a NEW session
    # We ignore the initial "Hello" handshake from the frontend for notifications to avoid spam
    if len(history) == 0 and user_query is not "Hello":
        tools.notify_user(f"New Visitor Session ({session_id[:8]}). Message: {user_query}")
    elif "[NEEDS_HUMAN]" in user_query: # Check if user is urgently asking for human? Unlikely, but good to have hooks.
        tools.notify_user(f"Urgent User Message: {user_query}")
    
    try:
        done = False
        attempts = 0
        max_attempts = 2 # Reduced from 3 to improve latency
        final_response = ""
        original_query = user_query

        # If this is the initial handshake "Hello" from logic, just pass it through 
        # but don't overthink it in the loop
        
        while not done and attempts < max_attempts:
            attempts += 1
            
            # 1. Primary Agent generates a draft
            draft = await agents.get_primary_response(user_query, history=history)
            
            # Check if human intervention is requested by the model
            if "[NEEDS_HUMAN]" in draft:
                tools.notify_user(f"System Requested Human Intervention. Query: {original_query}")
                tools.record_unknown_question(original_query)
                
                # Detect language (simple heuristic)
                if any(char in original_query.lower() for char in ['ü', 'ğ', 'ş', 'ı', 'ö', 'ç', 'merhaba', 'nasıl']):
                        final_response = "Talebini not ettim. Alperen bu konuda seninle bizzat iletişime geçecek."
                else:
                        final_response = "I have noted your inquiry and Alperen will get back to you personally regarding this matter."
                
                done = True
                break
                
            # 2. Critic Agent evaluates the draft
            # Only run critic if the draft is substantial, strictly speaking user might wait too long
            # For this version, we will trust the primary agent more, or use a faster critic.
            evaluation_raw = await agents.evaluate_response(original_query, draft)

            # Parse the evaluation
            try:
                # Clean up json markdown if present
                cleaned_eval = evaluation_raw.replace("```json", "").replace("```", "").strip()
                evaluation = json.loads(cleaned_eval)
            except json.JSONDecodeError:
                # If critic fails to output JSON, trust the primary agent
                final_response = draft
                done = True
                break

            if evaluation.get("score", 0) >= 8:
                final_response = draft
                done = True
            else:
                # Feedback loop: Update query with feedback for the next attempt
                user_query = f"Original Query: {original_query}\nFeedback from previous attempt: {evaluation.get('feedback')}\nImprove the response."
        
        # Fallback if loop finishes without high score
        if not final_response:
            final_response = draft # Use the last draft anyway

        # Update History
        sessions[session_id]["history"].append({"role": "user", "content": original_query})
        sessions[session_id]["history"].append({"role": "assistant", "content": final_response})
        
        return ChatResponse(response=final_response, session_id=session_id)

    except Exception as e:
        # Global error handler for the chat endpoint
        print(f"CRITICAL ERROR: {str(e)}")
        error_response = "I'm experiencing a temporary connection issue. Please try again in a moment."
        # Attempt to reply in Turkish if appropriate
        if any(char in original_query.lower() for char in ['ü', 'ğ', 'ş', 'ı', 'ö', 'ç', 'merhaba']):
             error_response = "Geçici bir bağlantı sorunu yaşıyorum. Lütfen biraz sonra tekrar deneyin."
             
        return ChatResponse(response=error_response, session_id=session_id)