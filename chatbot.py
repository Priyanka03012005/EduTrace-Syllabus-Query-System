"""
Chatbot implementation for EduTrace using Chatbase
"""

from flask import Flask, render_template, request, jsonify
import json
from knowledge_base import *

class EduTraceChatbot:
    def __init__(self):
        self.chatbase_id = "JqJscnY6sdNCYKyC6cnV-"
        self.base_url = "https://www.chatbase.co/chatbot-iframe/"
        self.iframe_url = f"{self.base_url}{self.chatbase_id}"
        
    def get_chatbot_embed_code(self):
        """Generate the HTML embed code for the Chatbase chatbot bubble"""
        return f"""
        <div id="chatbot-bubble" class="chatbot-bubble">
            <div id="chatbot-icon" class="chatbot-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="white"/>
                </svg>
            </div>
            <div id="chatbot-window" class="chatbot-window hidden">
                <div class="chatbot-header">
                    <span>EduTrace Assistant</span>
                    <button id="close-chatbot" class="close-button">×</button>
                </div>
                <div class="chatbot-iframe-container">
                    <iframe
                        src="{self.iframe_url}"
                        width="100%"
                        height="100%"
                        frameborder="0"
                        style="border: none;"
                        allow="microphone"
                        loading="eager"
                        onload="console.log('Chatbot iframe loaded successfully')"
                    ></iframe>
                </div>
            </div>
        </div>
        """
    
    def process_user_query(self, query):
        """Process user queries and return appropriate responses"""
        # Convert knowledge base to a format suitable for Chatbase
        knowledge_data = {
            "system_info": SYSTEM_INFO,
            "features": FEATURES,
            "faq": FAQ,
            "user_guide": USER_GUIDE,
            "error_messages": ERROR_MESSAGES,
            "system_commands": SYSTEM_COMMANDS
        }
        
        # Return the knowledge data for Chatbase to use
        return json.dumps(knowledge_data)

# Initialize Flask app
app = Flask(__name__)
chatbot = EduTraceChatbot()

@app.route('/')
def home():
    """Render the home page with the chatbot bubble"""
    try:
        return render_template('chatbot.html', 
                            chatbot_embed=chatbot.get_chatbot_embed_code())
    except Exception as e:
        return f"Error loading chatbot: {str(e)}", 500

@app.route('/api/chatbot/knowledge', methods=['GET'])
def get_knowledge_base():
    """API endpoint to provide knowledge base data to Chatbase"""
    try:
        return jsonify({
            "system_info": SYSTEM_INFO,
            "features": FEATURES,
            "faq": FAQ,
            "user_guide": USER_GUIDE,
            "error_messages": ERROR_MESSAGES,
            "system_commands": SYSTEM_COMMANDS
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 