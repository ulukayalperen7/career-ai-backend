import os
import json
import requests
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

"""
This module handles external API calls and side effects.
It implements the Notification and Alert mechanisms.
"""

def notify_user(message: str) -> str:
    """
    Sends a real push notification to Alperen via Pushover API.
    """
    # Fetching credentials from .env for security
    user_key = os.getenv("PUSHOVER_USER")
    token = os.getenv("PUSHOVER_TOKEN")
    url = "https://api.pushover.net/1/messages.json"

    if not user_key or not token:
        # Fallback if keys are missing
        print(f"\n[INTERNAL LOG] Notification keys missing. Content: {message}\n")
        return json.dumps({"status": "error", "reason": "API keys not found"})

    payload = {
        "user": user_key,
        "token": token,
        "message": message,
        "title": "Career AI Assistant"
    }

    try:
        # Making the actual API call to Pushover
        response = requests.post(url, data=payload)
        response.raise_for_status()
        return json.dumps({"status": "success", "response": "Notification pushed to mobile."})
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)})


def record_unknown_question(question: str) -> str:
    """
    Logs complex questions and alerts Alperen for manual intervention.
    """
    alert_msg = f"⚠️ ACTION REQUIRED: I encountered a question I couldn't answer: '{question}'"
    
    # We use the notification tool to alert you immediately
    notify_user(alert_msg)
    
    return json.dumps({
        "status": "human_intervention_requested",
        "message": "The question has been recorded and Alperen has been notified."
    })