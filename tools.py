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
    Sends a real push notification to Alperen via Telegram Bot API (or Pushover as fallback).
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_token = os.getenv("PUSHOVER_TOKEN")

    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"🤖 Career AI Assistant:\n\n{message}"
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return json.dumps({"status": "success", "response": "Notification pushed via Telegram."})
        except Exception as e:
            logging.error(f"Telegram Notification failed: {e}")
            return json.dumps({"status": "failed", "error": str(e)})
            
    elif pushover_user and pushover_token:
        # Legacy Pushover Support
        url = "https://api.pushover.net/1/messages.json"
        payload = {
            "user": pushover_user,
            "token": pushover_token,
            "message": message,
            "title": "Career AI Assistant"
        }
        try:
            response = requests.post(url, data=payload, timeout=5)
            response.raise_for_status()
            return json.dumps({"status": "success", "response": "Notification pushed via Pushover."})
        except Exception as e:
            logging.error(f"Pushover Notification failed: {e}")
            return json.dumps({"status": "failed", "error": str(e)})

    else:
        logging.warning(f"Notification keys missing. Content: {message}")
        return json.dumps({"status": "error", "reason": "API keys not found"})


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