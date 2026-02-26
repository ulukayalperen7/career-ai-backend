import os
import json
import requests
import asyncio
import logging
import csv
from datetime import datetime
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
    Attempts to be non-blocking if called from async context by using a thread, 
    but strictly this is a synchronous function.
    """
    # Fetching credentials from .env for security
    user_key = os.getenv("PUSHOVER_USER")
    token = os.getenv("PUSHOVER_TOKEN")
    url = "https://api.pushover.net/1/messages.json"

    if not user_key or not token:
        # Fallback if keys are missing
        logging.warning(f"Notification keys missing. Content: {message}")
        return json.dumps({"status": "error", "reason": "API keys not found"})

    payload = {
        "user": user_key,
        "token": token,
        "message": message,
        "title": "Career AI Assistant"
    }

    try:
        # Making the actual API call to Pushover
        # Use a short timeout to prevent hanging
        response = requests.post(url, data=payload, timeout=5)
        response.raise_for_status()
        return json.dumps({"status": "success", "response": "Notification pushed to mobile."})
    except Exception as e:
        logging.error(f"Notification failed: {e}")
        return json.dumps({"status": "failed", "error": str(e)})


def record_unknown_question(question: str) -> str:
    """
    Logs complex questions to a CSV file and alerts Alperen.
    """
    # 1. Log to file (persistent log for analysis)
    try:
        log_file = "unknown_questions.csv"
        timestamp = datetime.now().isoformat()
        
        # Check if file exists to write header
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Question'])
            writer.writerow([timestamp, question])
    except Exception as e:
        logging.error(f"Failed to log unknown question: {e}")

    # 2. Alert Alperen
    alert_msg = f"⚠️ ACTION REQUIRED: Unknown interaction reported: '{question}'"
    
    # We use the notification tool to alert you immediately
    return notify_user(alert_msg)