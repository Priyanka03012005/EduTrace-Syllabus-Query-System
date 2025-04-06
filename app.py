import os
import csv
import glob
import re
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from youtube_utils import get_youtube_links
from functools import wraps

YOUTUBE_API_KEY = "AIzaSyAGtjDc-6-oHbIb_ChhozbOtTrnUaHTo9s"

app = Flask(__name__)
app.secret_key = "edutracesecretsyllabus123"  # Added secret key for sessions
UPLOAD_FOLDER = 'data/pdf'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for uploads

# Create a data folder to store user information
USER_DATA_FOLDER = 'data/users'
os.makedirs(USER_DATA_FOLDER, exist_ok=True)
USER_DB_PATH = os.path.join(USER_DATA_FOLDER, 'users.csv')

# Create users.csv if it doesn't exist
if not os.path.exists(USER_DB_PATH):
    with open(USER_DB_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'password', 'role'])
        # Add a default admin user
        writer.writerow(['admin', generate_password_hash('admin123'), 'admin'])

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.pdf')

def is_valid_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            # Check if we can read the PDF successfully
            if pdf.pages and len(pdf.pages) > 0:
                return True
    except Exception as e:
        print(f"Error validating PDF: {e}")
    return False

def clear_data_folder():
    for f in glob.glob('data/*'):
        if os.path.isfile(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting file {f}: {e}")
        elif os.path.isdir(f) and not f.endswith('/users'):
            for subf in glob.glob(f + '/*'):
                try:
                    os.remove(subf)
                except Exception as e:
                    print(f"Error deleting file {subf}: {e}")

def pdf_to_csv(pdf_path, csv_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)
                    df.fillna("", inplace=True)
                    data.append(df)

    if data:
        final_df = pd.concat(data, ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"✅ CSV file saved at: {csv_path}")
        return True
    else:
        print("❌ No tables found in the PDF.")
        return False

def extract_course_and_modules(csv_path, output_path):
    df = pd.read_csv(csv_path, header=None)
    extracted_data = []
    current_course = None
    capture_modules = False

    for i, row in df.iterrows():
        row_values = row.astype(str).tolist()
        row_text = " ".join(row_values).lower()

        if "course code" in row_text and "course title" in row_text and "credit" in row_text:
            if i + 1 < len(df):
                current_course = df.iloc[i + 1, 1]
                capture_modules = False

        elif current_course and ("module" in row_text and "content" in row_text):
            capture_modules = True
            continue

        elif "textbook" in row_text:
            capture_modules = False

        elif capture_modules:
            extracted_data.append([current_course] + row_values)

    if extracted_data:
        final_df = pd.DataFrame(extracted_data, columns=["Course Title"] + [f"Column {i}" for i in range(len(extracted_data[0]) - 1)])
        final_df.to_csv(output_path, index=False)
        print(f"✅ Data extracted and saved to {output_path}")
        return True
    else:
        print("❌ No course modules found.")
        return False

def check_syllabus(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            content = "\n".join(["\n".join(row) for row in reader])
            target = "Third Year Engineering\n( Computer Engineering)"
            if target not in content:
                print("❌ PDF processed is not Third Year Engineering syllabus.")
                return False
            else:
                print("✅ Syllabus check passed.")
                return True
    except Exception as e:
        print(f"Error checking syllabus: {e}")
        return False

def process_pdf_data(csv_input_path, csv_output_path):
    try:
        df = pd.read_csv(csv_input_path)
        if df.empty:
            print("❌ Empty CSV file.")
            return False
            
        df = df.iloc[:, :6]
        df['Merged Column'] = df['Column 2'].combine_first(df['Column 3'])
        df.drop(columns=['Column 2', 'Column 3', 'Column 4'], inplace=True)

        df.rename(columns={
            'Course Title': 'Subject',
            'Column 0': 'Module no',
            'Merged Column': 'Module content',
            'Column 1': 'Replacement'
        }, inplace=True)

        df["Module no"] = pd.to_numeric(df["Module no"], errors="coerce")

        df["Module content"] = df.apply(
            lambda row: row["Replacement"] if re.fullmatch(r"\d+", str(row["Module content"])) else row["Module content"],
            axis=1
        )

        df.drop(columns=["Replacement"], inplace=True)
        df = df[~df.duplicated(subset=["Subject", "Module no"], keep="first")]
        df = df[df["Module no"] <= 6]
        
        # Clean up module content for better matching
        df["Module content"] = df["Module content"].astype(str).apply(lambda x: x.strip())
        
        df.to_csv(csv_output_path, index=False)
        print(f"✅ Final processed data saved to {csv_output_path}")
        return True
    except Exception as e:
        print(f"Error processing PDF data: {e}")
        return False

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_subject_module(question, df):
    question = preprocess_text(question)
    best_match = None
    max_score = 0
    question_words = set(question.split())
    
    # Create weighted keywords from question
    word_weights = {}
    for word in question_words:
        # Skip common short words
        if len(word) <= 3 or word in ["and", "the", "for", "what", "how", "when", "who", "why"]:
            continue
        
        # Higher weight for longer words (likely more specific terms)
        if len(word) >= 8:
            word_weights[word] = 3
        elif len(word) >= 6:
            word_weights[word] = 2
        else:
            word_weights[word] = 1
    
    # If we have very few keywords, try to use all words
    if len(word_weights) < 2:
        for word in question_words:
            if word not in word_weights and len(word) > 2:
                word_weights[word] = 0.5
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        try:
            subject = preprocess_text(str(row["Subject"]))
            module_content = preprocess_text(str(row["Module content"]))
            
            # Get words from both subject and module content
            combined_text = subject + " " + module_content
            
            # Calculate score for this row
            score = 0
            exact_matches = 0
            
            for word, weight in word_weights.items():
                # Check for exact word matches (higher value)
                if word in combined_text.split():
                    score += weight * 2
                    exact_matches += 1
                # Check for partial matches (lower value)
                elif word in combined_text:
                    score += weight
            
            # Bonus for consecutive word matches (phrases)
            for i in range(len(question.split()) - 1):
                phrase = " ".join(question.split()[i:i+2])
                if phrase in combined_text:
                    score += 2
            
            # Apply a small bonus if the module number is 1 (typically intro/foundation)
            # For very basic questions
            if str(row["Module no"]) == "1" and len(word_weights) < 3:
                score += 0.5
                
            # If we have a better score than our previous best
            if score > max_score:
                max_score = score
                
                # Calculate confidence based on exact matches and total keywords
                if len(word_weights) > 0:
                    confidence = min(100, int((score / (sum(word_weights.values()) * 2)) * 100))
                else:
                    confidence = 0
                    
                best_match = {
                    "subject": row["Subject"],
                    "module_no": row["Module no"],
                    "module_name": row["Module content"],
                    "confidence": confidence,
                    "score": score  # For debugging
                }
        except Exception as e:
            print(f"Error matching row: {e}")
            continue

    # If no good match is found
    if best_match is None or max_score < 1:
        return {
            "subject": "Out of Syllabus",
            "module_no": "N/A",
            "module_name": "No relevant module found.",
            "confidence": 0
        }
    
    return best_match

# User authentication functions
def get_user(username):
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row and row[0] == username:
                    return {"username": row[0], "password": row[1], "role": row[2]}
    return None

def add_user(username, password, role="user"):
    with open(USER_DB_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, generate_password_hash(password), role])

# Add a login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Clear user data when they log in
def clear_user_data():
    if os.path.exists("data/data.csv"):
        os.remove("data/data.csv")
    if os.path.exists("data/output.csv"):
        os.remove("data/output.csv")
    for f in glob.glob('data/pdf/*'):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting file {f}: {e}")
    # Clear the current file info from session
    if "current_file" in session:
        session.pop("current_file", None)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    show_input = False
    youtube_links = []
    question = None

    # Check if user is logged in
    is_logged_in = 'user_id' in session

    if request.method == "POST":
        # If trying to ask a question but not logged in, redirect to login
        if "question" in request.form and not is_logged_in:
            return redirect(url_for('login', next=request.url))
            
        if "pdf" in request.files and request.files["pdf"].filename:
            pdf = request.files["pdf"]
            if not allowed_file(pdf.filename):
                error = "❌ Please upload a PDF file - only PDF files are accepted"
            else:
                try:
                    filename = secure_filename(pdf.filename)
                    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    pdf.save(pdf_path)
                    
                    # Validate the PDF file
                    if not is_valid_pdf(pdf_path):
                        error = "❌ Invalid or corrupted PDF file. Please upload a valid PDF."
                        os.remove(pdf_path)
                    else:
                        # Clear previous data but keep the PDF
                        clear_data_folder()
                        # Re-save the PDF
                        pdf.seek(0)
                        pdf.save(pdf_path)
                        
                        # Process the PDF
                        if not pdf_to_csv(pdf_path, "data/output.csv"):
                            error = "❌ Could not extract tables from PDF. Please ensure it's a valid syllabus."
                        elif not check_syllabus("data/output.csv"):
                            error = "❌ Not a valid Third Year Computer Engineering syllabus. Please upload the correct syllabus."
                        elif not extract_course_and_modules("data/output.csv", "data/data.csv"):
                            error = "❌ Could not extract course modules from PDF. Please check the format."
                        elif not process_pdf_data("data/data.csv", "data/data.csv"):
                            error = "❌ Error processing data from PDF. Please try again."
                        else:
                            show_input = True
                            # Add success message after successful upload
                            session["file_upload_success"] = f"✅ File '{filename}' uploaded and processed successfully! Now you can ask questions about the syllabus."
                            # Store the current file name
                            session["current_file"] = filename
                            
                except Exception as e:
                    error = f"❌ Error processing PDF: {str(e)}"
                    print(error)

        elif "question" in request.form and request.form["question"] and is_logged_in:
            question = request.form["question"]
            try:
                if not os.path.exists("data/data.csv"):
                    error = "❌ Please upload a valid syllabus PDF first."
                else:
                    df = pd.read_csv("data/data.csv")
                    if df.empty:
                        error = "❌ No data found in syllabus. Please upload a valid syllabus PDF."
                    else:
                        result = detect_subject_module(question, df)
                        if result and result["subject"] != "Out of Syllabus":
                            youtube_links = get_youtube_links(result["module_name"], YOUTUBE_API_KEY)
                        show_input = True
            except Exception as e:
                error = f"❌ Error detecting module: {str(e)}"
                print(error)

    # Check if data.csv exists to determine if we should show the question input
    if os.path.exists("data/data.csv") and is_logged_in:
        show_input = True
    elif os.path.exists("data/data.csv") and not is_logged_in:
        # PDF is processed but user is not logged in
        error = "Please log in to ask questions and detect modules"
        show_input = False

    # Pass the current file name to template if it exists
    current_file = session.get("current_file", None)

    return render_template("index.html", result=result, error=error, show_input=show_input, 
                            youtube_links=youtube_links, question=question, is_logged_in=is_logged_in,
                            current_file=current_file)

@app.route("/module_detect", methods=["GET", "POST"])
@login_required
def module_detect():
    result = None
    error = None
    show_input = False
    youtube_links = []
    question = None

    if request.method == "POST":
        if "pdf" in request.files and request.files["pdf"].filename:
            pdf = request.files["pdf"]
            if not allowed_file(pdf.filename):
                error = "❌ Please upload a PDF file - only PDF files are accepted"
            else:
                try:
                    filename = secure_filename(pdf.filename)
                    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    pdf.save(pdf_path)
                    
                    # Validate the PDF file
                    if not is_valid_pdf(pdf_path):
                        error = "❌ Invalid or corrupted PDF file. Please upload a valid PDF."
                        os.remove(pdf_path)
                    else:
                        # Clear previous data but keep the PDF
                        clear_data_folder()
                        # Re-save the PDF
                        pdf.seek(0)
                        pdf.save(pdf_path)
                        
                        # Process the PDF
                        if not pdf_to_csv(pdf_path, "data/output.csv"):
                            error = "❌ Could not extract tables from PDF. Please ensure it's a valid syllabus."
                        elif not check_syllabus("data/output.csv"):
                            error = "❌ Not a valid Third Year Computer Engineering syllabus. Please upload the correct syllabus."
                        elif not extract_course_and_modules("data/output.csv", "data/data.csv"):
                            error = "❌ Could not extract course modules from PDF. Please check the format."
                        elif not process_pdf_data("data/data.csv", "data/data.csv"):
                            error = "❌ Error processing data from PDF. Please try again."
                        else:
                            show_input = True
                            # Add success message after successful upload
                            session["file_upload_success"] = f"✅ File '{filename}' uploaded and processed successfully! Now you can ask questions about the syllabus."
                except Exception as e:
                    error = f"❌ Error processing PDF: {str(e)}"
                    print(error)

        elif "question" in request.form and request.form["question"]:
            question = request.form["question"]
            try:
                if not os.path.exists("data/data.csv"):
                    error = "❌ Please upload a valid syllabus PDF first."
                else:
                    df = pd.read_csv("data/data.csv")
                    if df.empty:
                        error = "❌ No data found in syllabus. Please upload a valid syllabus PDF."
                    else:
                        result = detect_subject_module(question, df)
                        if result and result["subject"] != "Out of Syllabus":
                            youtube_links = get_youtube_links(result["module_name"], YOUTUBE_API_KEY)
                        show_input = True
            except Exception as e:
                error = f"❌ Error detecting module: {str(e)}"
                print(error)

    # Check if data.csv exists
    if os.path.exists("data/data.csv"):
        show_input = True
    
    # Pass the current file name to template if it exists
    current_file = session.get("current_file", None)

    return render_template("module_detection.html", result=result, error=error, show_input=show_input, 
                            youtube_links=youtube_links, question=question, is_logged_in=True,
                            current_file=current_file)

@app.route("/how")
def how():
    is_logged_in = 'user_id' in session
    return render_template("how_it_works.html", is_logged_in=is_logged_in)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_page = request.args.get('next', url_for('index'))
    
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user(username)
        
        if user and check_password_hash(user["password"], password):
            # Clear previous user data on login
            clear_user_data()
            
            session["user_id"] = username
            session["role"] = user["role"]
            session["session_message"] = "Welcome back! Previous data has been cleared. Please upload a new syllabus."
            
            return redirect(next_page)
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        if password != confirm_password:
            error = "Passwords do not match"
        elif get_user(username):
            error = "Username already exists"
        else:
            # Add the user to the database
            add_user(username, password)
            
            # Clear any previous data
            clear_user_data()
            
            # Set session variables
            session["user_id"] = username
            session["role"] = "user"
            session["session_message"] = "Account created successfully! You can now upload a syllabus."
            
            return redirect(url_for("index"))
    
    return render_template("signup.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# Add a before_request function to clear the session message after it's been shown once
@app.before_request
def clear_session_message():
    if 'session_message' in session and request.endpoint not in ['login', 'signup']:
        if request.method == 'GET':  # Only clear on GET requests to avoid clearing during form submission
            session.pop('session_message', None)
    
    # Also handle file upload success message
    if 'file_upload_success' in session and request.method == 'GET' and request.endpoint not in ['index', 'module_detect']:
        session.pop('file_upload_success', None)

if __name__ == "__main__":
    app.run(debug=True)

