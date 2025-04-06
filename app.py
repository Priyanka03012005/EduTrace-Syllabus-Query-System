import os
import csv
import glob
import re
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
from youtube_utils import get_youtube_links

YOUTUBE_API_KEY = "AIzaSyAGtjDc-6-oHbIb_ChhozbOtTrnUaHTo9s"

app = Flask(__name__)
UPLOAD_FOLDER = 'data/pdf'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.pdf')

def clear_data_folder():
    for f in glob.glob('data/*'):
        if os.path.isfile(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting file {f}: {e}")
        elif os.path.isdir(f):
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
    else:
        print("❌ No tables found in the PDF.")

    extract_course_and_modules(csv_path, "data/data.csv")

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

def check_syllabus(filepath):
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

def process_pdf_data(csv_input_path, csv_output_path):
    df = pd.read_csv(csv_input_path)
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
    df.to_csv(csv_output_path, index=False)
    print(f"✅ Final processed data saved to {csv_output_path}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def detect_subject_module(question, df):
    question = preprocess_text(question)
    best_match = None
    max_matches = 0

    for _, row in df.iterrows():
        module_content = preprocess_text(str(row["Module content"]))
        module_words = set(module_content.split())
        question_words = set(question.split())
        common_words = module_words.intersection(question_words)

        if len(common_words) > max_matches:
            max_matches = len(common_words)
            best_match = {
                "subject": row["Subject"],
                "module_no": row["Module no"],
                "module_name": row["Module content"]
            }

    return best_match if best_match else {
        "subject": "Out of Syllabus",
        "module_no": "N/A",
        "module_name": "No relevant module found."
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    show_input = False

    if request.method == "POST":
        if "pdf" in request.files and request.files["pdf"].filename.endswith(".pdf"):
            pdf = request.files["pdf"]
            clear_data_folder()
            filename = secure_filename(pdf.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(pdf_path)
            pdf_to_csv(pdf_path, "data/output.csv")

            if not check_syllabus("data/output.csv"):
                error = "❌ Not a valid Third Year Computer Engineering syllabus"
            else:
                extract_course_and_modules("data/output.csv", "data/data.csv")
                process_pdf_data("data/data.csv", "data/data.csv")
                show_input = True

        elif "question" in request.form:
            question = request.form["question"]
            try:
                df = pd.read_csv("data/data.csv")
                result = detect_subject_module(question, df)
                youtube_links = []  # Initialize before any condition
                if result and result["subject"] != "Out of Syllabus":
                     youtube_links = get_youtube_links(result["module_name"], YOUTUBE_API_KEY)
                return render_template("index.html", result=result, error=error, show_input=show_input, youtube_links=youtube_links)

                show_input = True
            except Exception as e:
                error = "❌ Please upload a valid syllabus PDF first."
                print(e)

    return render_template("index.html", result=result, error=error, show_input=show_input)

if __name__ == "__main__":
    app.run(debug=True)

