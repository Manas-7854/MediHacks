# importing required classes 
from pypdf import PdfReader 

# Open the PDF file
file_path = '/Users/siddharthshukla/Library/CloudStorage/OneDrive-ManipalUniversityJaipur/Kaam Dhandha/ML/Projects/MediHacks/Depression_RAG_Model/us_census/sodapdf-converted.pdf'
pdf_file = open(file_path, 'rb')

# Initialize PDF reader
pdf_reader = PdfReader(pdf_file)

# Extract text from each page
text = ""
for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    text += page.extract_text()

# Close the PDF file
pdf_file.close()

# Count the words
word_count = len(text.split())

print(f"The number of words in the PDF is: {word_count}")
