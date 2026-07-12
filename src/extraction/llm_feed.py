import json
import fitz
from pathlib import Path

FORMULA_KEYWORDS = ["=", "∑", "∫", "λ", "σ", "μ", "L(", "f(", "g(", "softmax", "logits"]


# ---------------------------------------------------------
# Load semantic blocks
# ---------------------------------------------------------
def load_blocks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Detect formulas in text
# ---------------------------------------------------------
def contains_formula(text):
    return any(k in text for k in FORMULA_KEYWORDS)


# ---------------------------------------------------------
# Filter blocks to reduce token usage
# ---------------------------------------------------------
def filter_blocks(blocks):
    filtered = []

    for blk in blocks:
        has_visual = blk["images"] or blk["drawings"]
        has_formula = contains_formula(blk["text"])

        if has_visual or has_formula:
            filtered.append(blk)

    return filtered


# ---------------------------------------------------------
# Extract only the PDF page that matters
# ---------------------------------------------------------
def extract_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap()
    img_path = f"data/processed/page_{page_number}.png"
    pix.save(img_path)
    return img_path


# ---------------------------------------------------------
# Build LLM payload
# ---------------------------------------------------------
def build_llm_payload(block, pdf_page_image):
    return {
        "page": block["page"],
        "block_id": block["block_id"],
        "text": block["text"],
        "images": block["images"],
        "drawings": block["drawings"],
        "pdf_page_image": pdf_page_image
    }


# ---------------------------------------------------------
# LLM extraction (multi-modal)
# ---------------------------------------------------------
def extract_metadata_from_block(llm, payload, schema):
    prompt = f"""
You are an expert in deep learning reproducibility.

Extract metadata according to this schema:
{json.dumps(schema, indent=2)}

Semantic Block:
- Page: {payload['page']}
- Block ID: {payload['block_id']}
- Text: {payload['text']}
- Images: {payload['images']}
- Drawings: {payload['drawings']}
- PDF Page Image: {payload['pdf_page_image']}

Your task:
1. Extract model architecture details.
2. Extract dataset details.
3. Extract preprocessing steps.
4. Extract hyperparameters.
5. Extract training setup.
6. Extract evaluation metrics.
7. Extract ablation study details.
8. Extract reproducibility notes.
9. Extract table data.
10. Extract figure interpretations.

Return JSON that strictly follows the schema.
"""

    response = llm(
    prompt,
    image_paths=[payload["pdf_page_image"]] + payload["images"] + payload["drawings"]
)


    return response


# ---------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------
def run_extraction(llm, json_path, pdf_path, schema_path):
    blocks = load_blocks(json_path)
    filtered_blocks = filter_blocks(blocks)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    results = []

    for blk in filtered_blocks:
        pdf_page_image = extract_pdf_page(pdf_path, blk["page"])
        payload = build_llm_payload(blk, pdf_page_image)
        metadata = extract_metadata_from_block(llm, payload, schema)
        results.append(metadata)

    out_path = "data/processed/metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return out_path
