import os
import json
import fitz  # PyMuPDF

PROCESSED_DIR = "data/processed"
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")

MIN_IMAGE_SIZE = 100  # ignore images smaller than 100x100 px


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. PARAGRAPH EXTRACTION
# ---------------------------------------------------------
def extract_paragraphs(page):
    """
    Extract paragraph-level text blocks from a PDF page.
    """
    paragraphs = []
    raw_blocks = page.get_text("dict")["blocks"]

    for block in raw_blocks:
        if block["type"] != 0:  # only text blocks
            continue

        lines = block.get("lines", [])
        paragraph_lines = []
        font_sizes = []

        for line in lines:
            spans = line.get("spans", [])
            line_parts = []

            for span in spans:
                text = span.get("text", "").strip()
                if text:
                    line_parts.append(text)
                    font_sizes.append(span.get("size", 0))

            if line_parts:
                paragraph_lines.append(" ".join(line_parts))

        if paragraph_lines:
            paragraphs.append({
                "bbox": block.get("bbox"),
                "text": " ".join(paragraph_lines),
                "font_size": sum(font_sizes) / len(font_sizes) if font_sizes else None
            })

    return paragraphs


# ---------------------------------------------------------
# 2. IMAGE EXTRACTION (Raster Images)
# ---------------------------------------------------------
def extract_raster_images(page, doc, page_num):
    """
    Extract raster images using page.get_images().
    Filters out small images (<100x100 px).
    """
    visuals = []

    for img in page.get_images(full=True):
        xref = img[0]

        try:
            pix = fitz.Pixmap(doc, xref)
        except Exception:
            continue

        # Filter small images
        if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
            continue

        img_name = f"page{page_num}_img_{xref}.png"
        img_path = os.path.join(IMAGE_DIR, img_name)

        if pix.n < 5:
            pix.save(img_path)
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(img_path)

        visuals.append({
            "type": "image",
            "path": img_path,
            "bbox": None  # raster images don't have block bbox
        })

    return visuals


# ---------------------------------------------------------
# 3. DRAWING / VECTOR EXTRACTION
# ---------------------------------------------------------
def extract_vector_drawings(page, page_num):
    """
    Rasterize vector drawings (equations, charts, diagrams).
    """
    visuals = []
    raw_blocks = page.get_text("dict")["blocks"]

    for idx, block in enumerate(raw_blocks):
        if block["type"] != 2:  # drawing/vector
            continue

        bbox = block.get("bbox")
        try:
            pix = page.get_pixmap(clip=bbox)
        except Exception:
            continue

        # Filter small drawings
        if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
            continue

        img_name = f"page{page_num}_draw_block{idx}.png"
        img_path = os.path.join(IMAGE_DIR, img_name)
        pix.save(img_path)

        visuals.append({
            "type": "drawing",
            "path": img_path,
            "bbox": bbox
        })

    return visuals


# ---------------------------------------------------------
# 4. ATTACH VISUALS TO PARAGRAPHS
# ---------------------------------------------------------
def attach_visuals(paragraphs, visuals):
    """
    Attach images/drawings to nearest paragraph based on vertical proximity.
    """
    semantic_blocks = []

    for p in paragraphs:
        semantic_blocks.append({
            "text": p["text"],
            "bbox": p["bbox"],
            "font_size": p["font_size"],
            "images": [],
            "drawings": [],
            "captions": []
        })

    for vis in visuals:
        if vis["bbox"] is None:
            # raster images have no bbox → attach to nearest paragraph center
            closest_idx = 0
            semantic_blocks[closest_idx]["images"].append(vis["path"])
            continue

        vx0, vy0, vx1, vy1 = vis["bbox"]
        v_center = (vy0 + vy1) / 2

        closest_idx = None
        closest_dist = float("inf")

        for i, sb in enumerate(semantic_blocks):
            tx0, ty0, tx1, ty1 = sb["bbox"]
            t_center = (ty0 + ty1) / 2
            dist = abs(v_center - t_center)

            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        if closest_idx is not None:
            if vis["type"] == "image":
                semantic_blocks[closest_idx]["images"].append(vis["path"])
            else:
                semantic_blocks[closest_idx]["drawings"].append(vis["path"])

    return semantic_blocks


# ---------------------------------------------------------
# 5. FULL PDF PARSING
# ---------------------------------------------------------
def extract_semantic_blocks(pdf_path):
    ensure_dirs()
    doc = fitz.open(pdf_path)
    all_blocks = []

    for page_num, page in enumerate(doc, start=1):
        paragraphs = extract_paragraphs(page)
        raster_images = extract_raster_images(page, doc, page_num)
        vector_drawings = extract_vector_drawings(page, page_num)

        visuals = raster_images + vector_drawings
        semantic_blocks = attach_visuals(paragraphs, visuals)

        for idx, sb in enumerate(semantic_blocks):
            all_blocks.append({
                "page": page_num,
                "block_id": idx,
                "type": "semantic_block",
                "text": sb["text"],
                "images": sb["images"],
                "drawings": sb["drawings"],
                "captions": sb["captions"],
                "bbox": sb["bbox"],
                "font_size": sb["font_size"]
            })

    return all_blocks


# ---------------------------------------------------------
# 6. SAVE OUTPUT
# ---------------------------------------------------------
def save_as_text(pdf_path, blocks):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out = os.path.join(PROCESSED_DIR, f"{base}_blocks.txt")

    with open(out, "w", encoding="utf-8") as f:
        for blk in blocks:
            f.write(f"[PAGE {blk['page']} | BLOCK {blk['block_id']}]\n")
            f.write("TEXT:\n")
            f.write(blk["text"] + "\n\n")

            if blk["images"]:
                f.write("IMAGES:\n")
                for img in blk["images"]:
                    f.write(f"- {img}\n")
                f.write("\n")

            if blk["drawings"]:
                f.write("DRAWINGS:\n")
                for dr in blk["drawings"]:
                    f.write(f"- {dr}\n")
                f.write("\n")

            f.write("-" * 40 + "\n\n")

    return out


def save_as_json(pdf_path, blocks):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out = os.path.join(PROCESSED_DIR, f"{base}_blocks.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)

    return out


# ---------------------------------------------------------
# 7. PUBLIC API
# ---------------------------------------------------------
def parse_pdf(pdf_path):
    blocks = extract_semantic_blocks(pdf_path)
    text_path = save_as_text(pdf_path, blocks)
    json_path = save_as_json(pdf_path, blocks)
    return blocks, text_path, json_path
