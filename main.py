"""
Main entry point for AI Salon Receptionist
Run this file to start the interactive chat interface
"""

from app.ai_receptionist import chat_with_receptionist, demo_receptionist
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_receptionist()
    else:
        chat_with_receptionist()


