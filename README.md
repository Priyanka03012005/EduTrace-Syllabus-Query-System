# EduTrace - Syllabus Query System

EduTrace is an intelligent, AI-powered web platform that helps students effortlessly identify the subject, module number, and content of any syllabus-related question from their engineering curriculum.


## Features

- **PDF Syllabus Upload**: Upload and process Third Year Computer Engineering syllabus PDFs
- **File Validation**: Comprehensive PDF validation to ensure only valid syllabus files are processed
- **Question Analysis**: Ask questions to identify which module and subject they belong to
- **Confidence Scoring**: Shows match confidence to help evaluate accuracy of results
- **YouTube Integration**: Automatically fetch relevant educational videos for detected topics
- **User Authentication**: Login and signup functionality for personalized experience
- **File Status Indicator**: Clear visual feedback showing currently loaded file
- **Session Management**: User data is cleared on login for a personalized experience
- **Responsive Design**: Works across all devices and screen sizes

## Installation Requirements

```
pip install flask pdfplumber pandas werkzeug requests
```

## How to Run

1. Clone this repository
2. Install the required packages
3. Run the Flask application:

```
python app.py
```

4. Open your browser and navigate to http://127.0.0.1:5000/

## Default Login

- **Username**: admin
- **Password**: admin123

## PDF Requirements

- Must be a valid PDF file (not corrupted)
- Must contain Third Year Computer Engineering syllabus content
- Should have proper table structure for module extraction

## File Structure

- `app.py`: Main Flask application
- `youtube_utils.py`: YouTube API integration utilities
- `templates/`: HTML templates
- `static/css/`: CSS stylesheets
- `data/`: Data storage directory
  - `pdf/`: Uploaded PDF files
  - `users/`: User information

## How It Works

1. Upload the official Third Year Computer Engineering syllabus PDF
2. The system validates and processes the syllabus
3. Ask any syllabus-related question
4. The system detects the subject, module number, and content with confidence score
5. Related YouTube videos are displayed alongside the results

## User Experience Improvements

- **Clear File Feedback**: Users can always see which file is currently loaded
- **Enhanced Notifications**: Success messages for file uploads and login actions
- **User Data Isolation**: Each user's data is cleared on login to prevent cross-contamination
- **Improved Error Handling**: Comprehensive error messages guide users to correct actions

## Technologies Used

- Flask (Python web framework)
- pdfplumber (PDF parsing)
- Pandas (Data processing)
- HTML, CSS (Frontend)
- YouTube Data API v3 

## Recent Updates

- Added current file indicator to show which syllabus is loaded
- Fixed various UI bugs and improved responsive design 
