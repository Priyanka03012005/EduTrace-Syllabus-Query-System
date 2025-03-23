import os
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
from flask import Flask, render_template, request
from fuzzywuzzy import process
import openpyxl
from threading import Thread

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
EXCEL_FILE = "syllabus_data.xlsx"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def async_extract_text(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    return text

def save_file_async(pdf):
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)
    thread = Thread(target=async_extract_text, args=(pdf_path,))
    thread.start()



# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to parse syllabus and extract course and module details
def parse_syllabus(text):
    lines = text.split("\n")
    courses = []
    current_course = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Identify Course Name (Assumption: Capitalized words with & or spaces)
        if line.istitle() or "&" in line:
            current_course = line
        elif current_course and line[0].isdigit():  # Identify Module (Assumption: Starts with number)
            module_number, module_name = line.split(".", 1)
            courses.append([current_course, module_number.strip(), module_name.strip()])

    return courses

# Function to save syllabus data to Excel
def save_to_excel(data):
    df = pd.DataFrame(data, columns=["Course Name", "Module Number", "Module Name"])
    df.to_excel(EXCEL_FILE, index=False)

# Function to search for the most relevant course and module
def search_syllabus(query):
    df = pd.read_excel(EXCEL_FILE)
    
    # Find the best course match
    best_course_match, _ = process.extractOne(query, df["Course Name"].unique())

    # Filter data for the matched course
    matched_df = df[df["Course Name"] == best_course_match]

    # Find the best module match
    best_module_match, _ = process.extractOne(query, matched_df["Module Name"].tolist())

    # Get the module number for the matched module
    module_number = matched_df[matched_df["Module Name"] == best_module_match]["Module Number"].values[0]

    return best_course_match, module_number, best_module_match

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "pdf" in request.files:
            pdf = request.files["pdf"]
            if pdf.filename.endswith(".pdf"):
                pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
                pdf.save(pdf_path)

                # Extract and parse syllabus
                text = extract_text_from_pdf(pdf_path)
                syllabus_data = parse_syllabus(text)
                save_to_excel(syllabus_data)

        elif "question" in request.form:
            question = request.form["question"]
            result = search_syllabus(question)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
