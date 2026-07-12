import gradio as gr
import json
import glob
import os

from src.injection.paper_parser import parse_pdf
from src.utils.file_utils import save_pdf
from src.extraction.llm_feed import run_extraction
from src.extraction.llm_models import get_llm


def handle_upload(file):
    try:
        file_path, message = save_pdf(file)

        blocks, text_path, json_path = parse_pdf(file_path)

        final_message = (
            f"{message}\n"
            f"Semantic blocks (text) saved to: {text_path}\n"
            f"Semantic blocks (JSON) saved to: {json_path}\n"
            f"Total semantic blocks: {len(blocks)}"
        )

        return final_message

    except Exception as e:
        return f"Error: {str(e)}"


def run_llm_extraction():
    # find latest blocks file
    blocks = sorted(glob.glob("data/processed/*_blocks.json"))
    if not blocks:
        return {"error": "No semantic blocks found. Upload a PDF first."}

    json_path = blocks[-1]

    # find latest PDF
    pdfs = sorted(glob.glob("data/raw/*.pdf"))
    if not pdfs:
        return {"error": "No PDF found in data/raw."}

    pdf_path = pdfs[-1]

    schema_path = "src/extraction/schema/reproducibility_schema.json"

    llm = get_llm()  # HuggingFace VLM

    metadata_path = run_extraction(llm, json_path, pdf_path, schema_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def view_metadata():
    path = "data/processed/metadata.json"
    if not os.path.exists(path):
        return {"error": "metadata.json not found. Run LLM extraction first."}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 📄 DL-KG Builder — Upload Research Paper")
        gr.Markdown("Upload a PDF research paper. It will be saved in `data/raw/`.")

        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        output = gr.Textbox(label="Status", interactive=False)

        upload_button = gr.Button("Upload")
        upload_button.click(fn=handle_upload, inputs=pdf_input, outputs=output)

        gr.Markdown("## 🔍 Run LLM Extraction (Local HuggingFace VLM)")

        extract_button = gr.Button("Run LLM Extraction")
        metadata_output = gr.JSON(label="Extracted Metadata")

        extract_button.click(fn=run_llm_extraction, inputs=None, outputs=metadata_output)

        gr.Markdown("## 📦 View Extracted Metadata")

        view_button = gr.Button("View Metadata")
        view_button.click(fn=view_metadata, inputs=None, outputs=metadata_output)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
