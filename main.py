from fastapi import FastAPI
from pydantic import BaseModel
import agents
import tools
import json

app = FastAPI(title="Career Assistant Agentic System")

# Memory Implementation: Sliding window buffer
conversation_history = []
MAX_HISTORY = 5

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def run_agent_system(request: ChatRequest):
    global conversation_history
    user_query = request.message
    
    # [Step 1: Notify that a new message arrived]
    tools.notify_user(f"New recruiter message: {user_query}")
    
    done = False
    attempts = 0
    max_attempts = 3
    final_response = ""
    evaluation_log = []
    
    # Keep original query for evaluation
    original_query = user_query

    while not done and attempts < max_attempts:
        attempts += 1
        
        try:
            # 1. Primary Agent generates a draft
            draft = agents.get_primary_response(user_query, history=conversation_history)
            
            # Check if human intervention is requested
            if "[NEEDS_HUMAN]" in draft:
                tools.record_unknown_question(original_query)
                final_response = "I have noted your inquiry and Alperen will get back to you personally regarding this matter."
                done = True
                break
                
            # 2. Critic Agent evaluates the draft
            evaluation_raw = agents.evaluate_response(original_query, draft)
            
            try:
                # Cleaning the LLM output to get valid JSON
                eval_clean = evaluation_raw.replace("```json", "").replace("```", "").strip()
                evaluation = json.loads(eval_clean)
            except Exception as e:
                print(f"JSON Parse Error: {e}, Raw: {evaluation_raw}")
                evaluation = {"score": 5, "feedback": "Failed to parse evaluation, retrying."}

            evaluation_log.append(evaluation)
            
            if evaluation.get("score", 0) >= 8:
                final_response = draft
                done = True
            else:
                # If score is low, the loop continues for another attempt
                user_query = f"{original_query} (Correction: {evaluation.get('feedback')})"
                
        except Exception as e:
            print(f"API Error: {e}")
            # Handle rate limits or other API errors gracefully
            if "429" in str(e) or "quota" in str(e).lower():
                final_response = "I am currently experiencing high traffic. Please try again in a few minutes."
            elif "503" in str(e) or "unavailable" in str(e).lower():
                final_response = "The AI model is currently experiencing high demand. Please try again in a few moments."
            else:
                final_response = "An unexpected error occurred while processing your request."
            
            evaluation_log.append({"error": str(e)})
            done = True
            break

    # [Step 3: Check for Unknown Questions / Thresholds]
    # If after max attempts we still fail, or it's a risky topic
    if not done and not final_response:
        tools.record_unknown_question(original_query)
        final_response = "I have noted your inquiry and Alperen will get back to you personally regarding this matter."

    # Update memory
    conversation_history.append(f"Recruiter: {original_query}")
    conversation_history.append(f"Agent: {final_response}")
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history = conversation_history[-(MAX_HISTORY * 2):]

    # [Step 4: Final notification to Alperen]
    tools.notify_user(f"Response sent to recruiter: {final_response}")

    return {
        "agent_response": final_response,
        "evaluation_history": evaluation_log,
        "attempts": attempts
    }