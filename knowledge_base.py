"""
Knowledge Base for EduTrace Chatbot
This file contains structured information about the EduTrace system for use in a chatbot.
"""

# System Information
SYSTEM_INFO = {
    "name": "EduTrace",
    "description": "An intelligent, AI-powered web platform that helps students identify syllabus-related information",
    "version": "1.0",
    "purpose": "To help students identify subject, module number, and content of syllabus-related questions"
}

# Features
FEATURES = [
    {
        "name": "PDF Syllabus Upload",
        "description": "Upload and process Third Year Computer Engineering syllabus PDFs",
        "requirements": "Must be a valid PDF file with proper table structure"
    },
    {
        "name": "Question Analysis",
        "description": "Analyze questions to identify which module and subject they belong to",
        "capabilities": [
            "Detect subject",
            "Identify module number",
            "Extract relevant content",
            "Provide confidence scores"
        ]
    },
    {
        "name": "YouTube Integration",
        "description": "Automatically fetch relevant educational videos for detected topics",
        "requirements": "Requires YouTube API key"
    },
    {
        "name": "User Management",
        "description": "Login and signup functionality for personalized experience",
        "features": [
            "User authentication",
            "Session management",
            "Personalized data storage"
        ]
    }
]

# Common Questions and Answers
FAQ = [
    {
        "question": "What types of PDFs can I upload?",
        "answer": "You can upload Third Year Computer Engineering syllabus PDFs. The PDF must be valid and contain proper table structure for module extraction."
    },
    {
        "question": "How do I get started?",
        "answer": "1. Create an account or login\n2. Upload a valid syllabus PDF\n3. Start asking questions about the syllabus content"
    },
    {
        "question": "What information can I get from the system?",
        "answer": "You can get information about:\n- Subject identification\n- Module number\n- Module content\n- Related YouTube videos\n- Confidence scores for matches"
    },
    {
        "question": "How accurate are the results?",
        "answer": "The system provides confidence scores to help you evaluate the accuracy of results. The accuracy depends on the quality of the uploaded syllabus and the clarity of your question."
    }
]

# Technical Requirements
TECHNICAL_INFO = {
    "dependencies": [
        "flask",
        "pdfplumber",
        "pandas",
        "werkzeug",
        "requests"
    ],
    "python_version": "3.6+",
    "database": "SQLite",
    "api_integrations": [
        "YouTube Data API v3"
    ]
}

# User Guide
USER_GUIDE = {
    "getting_started": [
        "Install required packages",
        "Run the Flask application",
        "Access the web interface at http://127.0.0.1:5000/",
        "Login with default credentials (admin/admin123) or create new account"
    ],
    "file_upload": [
        "Click on the upload button",
        "Select a valid syllabus PDF",
        "Wait for processing confirmation",
        "Start asking questions"
    ],
    "question_format": [
        "Ask questions in natural language",
        "Be specific about the topic or concept",
        "Include relevant keywords",
        "Example: 'What is the content of Module 3 in Computer Networks?'"
    ]
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_pdf": "The uploaded file must be a valid PDF containing Third Year Computer Engineering syllabus content.",
    "authentication": "Invalid username or password. Please try again.",
    "file_processing": "Error processing the PDF file. Please ensure it meets the requirements.",
    "api_error": "Error connecting to YouTube API. Please try again later."
}

# System Commands
SYSTEM_COMMANDS = {
    "help": "Display help information and available commands",
    "upload": "Upload a new syllabus PDF",
    "analyze": "Analyze a question against the current syllabus",
    "logout": "Logout from the current session",
    "clear": "Clear current session data"
} 