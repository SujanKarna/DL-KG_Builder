import os
from datetime import datetime


DATA_DIR = "data/raw"

def save_pdf(file):
    # Check if file exists
    if file is None:
        raise ValueError("No file provided for saving.")

    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.basename(file.name)
    filename = f"{timestamp}_{original_name}"
    file_path = os.path.join(DATA_DIR, filename)

    # Gradio gives a temp file path, not a file object
    temp_path = file.name

    # Copy file bytes from temp file to our destination
    with open(temp_path, "rb") as src:
        with open(file_path, "wb") as dst:
            dst.write(src.read())

    message = f"File saved to: {file_path}"
    
    return file_path, message