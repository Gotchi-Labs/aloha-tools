import pdfplumber
import os
import json
import re

# Define a regular expression pattern to identify Tax Map Keys
tax_map_key_pattern = re.compile(r'(\d-\d-\d{3}-\d{3}-\d{4})')

# Define a logging function for clean output messages
def log(message):
    print(f"[INFO] {message}")

# Function to parse a single line into a record, if it matches the TMK pattern
def parse_line_to_record(line):
    match = tax_map_key_pattern.search(line)
    if match:
        tmk = match.group(1)
        # Split the remainder of the line by whitespace to extract the data values
        values = line[match.end():].split()
        record = {
            'Tax Map Key': tmk,
            'PT': values[0],
            'Value': values[1],
            'Exempt': values[2],
            'Taxable': values[3],
            'Appeal Amount In Dispute': values[4] if len(values) > 4 else ''
        }
        return record
    return None

# Function to process each page of the PDF and extract records
def extract_data_from_page(page):
    log(f"Processing page {page.page_number}...")
    text = page.extract_text()
    if not text:
        log(f"No text found on page {page.page_number}.")
        return []
    
    # Initialize an empty list to hold the records found on the page
    records = []
    for line in text.split('\n'):
        record = parse_line_to_record(line)
        if record:
            records.append(record)
    return records

# Main function to process the entire PDF and compile data into a list
def main(pdf_path):
    extracted_data = []
    if not os.path.exists(pdf_path):
        log(f"File not found: {pdf_path}")
        return []

    with pdfplumber.open(pdf_path) as pdf:
        log(f"Opened PDF with {len(pdf.pages)} pages.")
        for page_number, page in enumerate(pdf.pages, start=1):
            # Process a limited number of pages for testing purposes
            if page_number > 100:
                log("Reached page limit of 100 for testing.")
                break
            # Extract data from the current page and add it to the overall data list
            page_data = extract_data_from_page(page)
            if page_data:
                extracted_data.extend(page_data)
                log(f"Extracted data from page {page.page_number}.")
            else:
                log(f"No data extracted from page {page.page_number}.")
        log(f"Extraction completed. Total pages processed: {page_number}")
    
    return extracted_data

# Entry point of the script
if __name__ == "__main__":
    pdf_filename = 'target.pdf'
    pdf_path = os.path.join(os.path.dirname(__file__), pdf_filename)
    log(f"Starting data extraction for {pdf_path}...")
    data = main(pdf_path)
    
    # Save the extracted data to a JSON file
    if data:
        json_filename = 'extracted_data.json'
        json_path = os.path.join(os.path.dirname(__file__), json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        log(f"Data extraction completed and saved to '{json_filename}'.")
    else:
        log("No data was extracted.")
