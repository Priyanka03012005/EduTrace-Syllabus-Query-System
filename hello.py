import pdfplumber
import pandas as pd
import re

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
        print(f"✅ Data extracted and saved to {output_path}")

# Example Usage
csv_input_path = "output.csv"  # Replace with actual CSV path
csv_output_path = "data.csv"
extract_course_and_modules(csv_input_path, csv_output_path)

# Process and clean extracted data
df = pd.read_csv('data.csv')
df = df.iloc[:, :6]

# Merge columns and clean up
df['Merged Column'] = df['Column 2'].combine_first(df['Column 3'])
df.drop(columns=['Column 2', 'Column 3', 'Column 4'], inplace=True)
df.rename(columns={
    'Course Title': 'Subject',
    'Column 0': 'Module no',
    'Merged Column': 'Module content',
    'Column 1': 'Replacement'
}, inplace=True)

# Convert "Module no" to numeric (coerce errors to NaN)
df["Module no"] = pd.to_numeric(df["Module no"], errors="coerce")

# Replace numeric module content with corresponding value from "Replacement"
df["Module content"] = df.apply(
    lambda row: row["Replacement"] if re.fullmatch(r"\d+", str(row["Module content"])) else row["Module content"],
    axis=1
)

# Drop the "Replacement" column
df.drop(columns=["Replacement"], inplace=True)

# Identify the first occurrence and keep it, remove subsequent duplicates
df["is_duplicate"] = df.duplicated(subset=["Subject", "Module no"], keep="first")

# Keep only non-duplicate entries
df = df[df["is_duplicate"] == False].drop(columns=["is_duplicate"])

# Keep only the first 6 modules per subject
df = df[df["Module no"] <= 6]

# Save the cleaned data
df.to_csv('data.csv', index=False)
print("✅ Final processed data saved to data.csv")
