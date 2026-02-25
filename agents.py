import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# 1. Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Load Alperen's profile once to use in prompts
with open("profile.txt", "r", encoding="utf-8") as f:
    PROFILE_CONTEXT = f.read()

def get_primary_response(user_message: str, history: list = None):
    """
    Primary Agent (Career Agent): Generates the first draft of the response.
    [Hoca Requirement: Primary Agent]
    """
    system_instruction = f"""
    You are Alperen Ulukaya's official Career AI Assistant. 
    Your goal is to represent him professionally to recruiters.
    
    Context about Alperen:
    {PROFILE_CONTEXT}
    
    Instructions:
    - Use a professional, concise, and polite tone.
    - If you are asked about salary, legal matters, or topics not in the profile, 
      you MUST reply EXACTLY with the phrase: [NEEDS_HUMAN]
    - For technical questions, answer based on Alperen's tech stack (Spring Boot, .NET, Angular, etc.).
    """
    
    # Memory Implementation: Include history if provided
    chat_context = ""
    if history:
        chat_context = "Previous Conversation:\n" + "\n".join(history) + "\n\n"
        
    full_message = chat_context + "Current Message: " + user_message
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        ),
        contents=full_message
    )
    return response.text

def evaluate_response(original_query: str, proposed_response: str):
    """
    Response Evaluator (Critic Agent): Scores the response.
    [Hoca Requirement: Response Evaluator / Self-Critic]
    """
    eval_prompt = f"""
    Evaluate the following AI-generated response based on these criteria:
    1. Professional Tone (1-10)
    2. Accuracy (1-10)
    3. Safety (No false claims) (1-10)
    
    Original Recruiter Query: {original_query}
    Proposed Response: {proposed_response}
    
    Return your evaluation in JSON format with exactly two keys: 'score' (integer average) and 'feedback' (string).
    """
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            temperature=0.1, # Low temp for strict evaluation
            response_mime_type="application/json" # Enforce JSON output to prevent 500 errors
        ),
        contents=eval_prompt
    )
    return response.text