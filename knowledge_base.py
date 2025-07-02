"""
Knowledge Base for EduTrace Chatbot
This file contains structured information about the EduTrace system for use in a chatbot.
"""

import json

# Load keywords from keyword.json
def load_keywords():
    try:
        with open('keyword.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading keywords: {e}")
        return {}

# System Information
SYSTEM_INFO = {
    "name": "EduTrace",
    "version": "1.0",
    "description": "Smart Question Categorization System for Educational Content",
    "purpose": "Help students and educators find relevant course content based on natural language questions"
}

# System Features
FEATURES = [
    {
        "name": "Smart Question Categorization",
        "description": "Uses AI to match your questions with relevant course modules",
        "how_to_use": "Simply type your question about any topic, and the system will find the most relevant module"
    },
    {
        "name": "Syllabus Processing",
        "description": "Upload and process PDF syllabi to extract course information",
        "how_to_use": "Upload a PDF syllabus file through the module detection page"
    },
    {
        "name": "YouTube Integration",
        "description": "Get relevant YouTube video recommendations for matched modules",
        "how_to_use": "After asking a question, relevant YouTube videos will be displayed"
    },
    {
        "name": "Confidence Scoring",
        "description": "See how confident the system is about its matches",
        "how_to_use": "Check the confidence scores displayed with each match"
    }
]

# Frequently Asked Questions
FAQ = [
    {
        "question": "How do I upload a syllabus?",
        "answer": "Go to the Module Detection page, click 'Choose File', select your PDF syllabus, and click 'Upload'"
    },
    {
        "question": "What types of questions can I ask?",
        "answer": "You can ask any question related to your course content. The system will find the most relevant module"
    },
    {
        "question": "How accurate is the system?",
        "answer": "The system uses AI to provide accurate matches. You'll see confidence scores to indicate match quality"
    },
    {
        "question": "Can I see my previous questions?",
        "answer": "Currently, the system processes questions in real-time and doesn't store history"
    },
    {
        "question": "What if my question doesn't match any module?",
        "answer": "The system will indicate 'Out of Syllabus' if no relevant match is found"
    }
]

# User Guide
USER_GUIDE = {
    "getting_started": [
        "1. Upload your syllabus PDF",
        "2. Wait for processing to complete",
        "3. Start asking questions about course content",
        "4. View matched modules and YouTube recommendations"
    ],
    "best_practices": [
        "Be specific in your questions",
        "Use natural language",
        "Focus on course-related topics",
        "Check confidence scores for match quality"
    ],
    "troubleshooting": [
        "If upload fails, check PDF format",
        "For no matches, try rephrasing your question",
        "If videos don't load, check your internet connection",
        "Contact support if issues persist"
    ]
}

# Error Messages
ERROR_MESSAGES = {
    "upload_error": "Error uploading syllabus. Please check file format and try again.",
    "processing_error": "Error processing syllabus. Please ensure it's a valid PDF.",
    "no_match": "No relevant module found. Try rephrasing your question.",
    "api_error": "Error connecting to services. Please try again later.",
    "invalid_syllabus": "Invalid syllabus format. Please upload a valid syllabus PDF."
}

# System Commands
SYSTEM_COMMANDS = {
    "help": "Show this help message",
    "features": "List all system features",
    "guide": "Show user guide",
    "faq": "Show frequently asked questions",
    "clear": "Clear current conversation",
    "about": "Show system information"
}

# Chatbot Responses
CHATBOT_RESPONSES = {
    "greeting": [
        "Hello! How can I help you with your course content today?",
        "Hi there! What would you like to know about your courses?",
        "Welcome! Ask me anything about your syllabus."
    ],
    "farewell": [
        "Goodbye! Feel free to come back with more questions.",
        "See you later! Don't hesitate to ask if you need help.",
        "Bye! Have a great learning experience!"
    ],
    "thanks": [
        "You're welcome! Is there anything else I can help with?",
        "Glad I could help! Feel free to ask more questions.",
        "Happy to assist! Let me know if you need anything else."
    ],
    "confused": [
        "I'm not sure I understand. Could you rephrase that?",
        "Could you provide more details about what you're looking for?",
        "I'm having trouble understanding. Could you try asking differently?"
    ]
}

# Load subject and module data from keyword.json
keyword_data = load_keywords()

# Subject-Specific Keywords
SUBJECT_KEYWORDS = {}
for subject, modules in keyword_data.items():
    # Collect all keywords from all modules for each subject
    all_keywords = []
    for module_keywords in modules.values():
        all_keywords.extend(module_keywords)
    SUBJECT_KEYWORDS[subject] = list(set(all_keywords))  # Remove duplicates

# Module-Specific Help
MODULE_HELP = {}
for subject, modules in keyword_data.items():
    for module_name, keywords in modules.items():
        MODULE_HELP[module_name] = f"Topics in {module_name} include: {', '.join(keywords[:5])}..."

# Learning Resources
LEARNING_RESOURCES = {
    "videos": "YouTube video recommendations are provided for matched modules",
    "textbooks": "Refer to your course textbooks for detailed information",
    "practice": "Try solving related problems to reinforce understanding",
    "discussion": "Engage in discussions with peers and instructors"
} 