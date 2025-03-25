import pdfplumber
import pandas as pd

def pdf_to_csv(pdf_path, csv_path):
    data = []  # Store extracted text data

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()  # Extract tables from the page
            
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)  # Convert table to DataFrame
                    df.fillna("", inplace=True)  # Replace None values with empty strings
                    data.append(df)

    # Combine all tables into one DataFrame
    if data:
        final_df = pd.concat(data, ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"✅ CSV file saved at: {csv_path}")
    else:
        print("❌ No tables found in the PDF.")

# Example usage
pdf_path = "sem.pdf"  # Replace with your PDF file path
csv_path = "output.csv"  # Output CSV file path
pdf_to_csv(pdf_path, csv_path)
import pandas as pd
import re

def extract_course_and_modules(csv_path, output_path):
    # Load CSV file without headers
    df = pd.read_csv(csv_path, header=None)

    extracted_data = []  # Store extracted course and module details
    current_course = None
    capture_modules = False  # Flag to start capturing module details

    for i, row in df.iterrows():
        row_values = row.astype(str).tolist()  # Convert row to list of strings
        row_text = " ".join(row_values).lower()  # Convert row to lowercase for easy search

        # Detect a Course Code, Course Title, and Credit in the same row
        if "course code" in row_text and "course title" in row_text and "credit" in row_text:
            if i + 1 < len(df):  # Ensure next row exists
                current_course = df.iloc[i + 1, 1]  # Extract Course Title
                capture_modules = False  # Reset module capture flag

        # Detect Module Header (start capturing module details)
        elif current_course and ("module" in row_text and "content" in row_text):
            capture_modules = True
            continue  # Skip this row, start capturing from the next

        # Stop capturing when "Textbook" is encountered
        elif "textbook" in row_text:
            capture_modules = False

        # Capture Module Details Until "Textbook" Appears
        elif capture_modules:
            extracted_data.append([current_course] + row_values)

    # Convert extracted data to DataFrame and save to CSV
    if extracted_data:
        final_df = pd.DataFrame(extracted_data, columns=["Course Title"] + [f"Column {i}" for i in range(len(extracted_data[0]) - 1)])
        final_df.to_csv(output_path, index=False)

# Example Usage
csv_input_path = "output.csv"  # Replace with actual CSV path
csv_output_path = "structured_output.csv"
extract_course_and_modules(csv_input_path, csv_output_path)

print(f"✅ Data extracted and saved to {csv_output_path}")
