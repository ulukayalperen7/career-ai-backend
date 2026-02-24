import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

"""
Docstrings for Homework Requirements:
This module defines the Primary Career Agent and the Critic Agent.
It handles the prompt engineering and the decision-making logic.
[Hoca Requirement: Primary Agent & Response Evaluator]
"""

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
      you MUST signal that you need human intervention.
    - For technical questions, answer based on Alperen's tech stack (Spring Boot, .NET, Angular, etc.).
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        ),
        contents=user_message
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
    
    Return your evaluation in JSON format with 'score' (average) and 'feedback'.
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=0.1), # Low temp for strict evaluation
        contents=eval_prompt
    )
    return response.text