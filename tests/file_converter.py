import PyPDF2
import os

def extract_text_from_pdf(pdf_path, txt_path):
    # Open the provided PDF file
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
        # Open a text file for writing the extracted text with UTF-8 encoding
        with open(txt_path, "w", encoding="utf-8") as text_file:
            # Iterate through each page in the PDF
            for page in reader.pages:
                # Extract text from the page
                text = page.extract_text()
                if text:  # Check if text is extracted
                    text_file.write(text)
    
    print(f"Text has been extracted and saved to {txt_path}")

def process_all_pdfs(pdf_folder, txt_folder):
    # Ensure the output folder exists
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    
    # Loop through all files in the directory
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            txt_path = os.path.join(txt_folder, filename.replace(".pdf", ".md"))
            extract_text_from_pdf(pdf_path, txt_path)

# Example usage
pdf_folder = "C:\\Rohan Workplace\\Rohan's Second Brain\\Dr. Reddys"  # Change to your folder path with PDFs
txt_folder = "C:\\Rohan Workplace\\Rohan's Second Brain\\Dr. Reddys"   # Change to your desired output folder path for text files
process_all_pdfs(pdf_folder, txt_folder)
