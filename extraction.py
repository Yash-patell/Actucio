import fitz  
import pdfplumber
from paddleocr import PaddleOCR
from PIL import Image
import io

# Initialize OCR (French + English)
# ocr = PaddleOCR(lang='fr')  # 'fr' = French, add 'en' if bilingual
ocr = PaddleOCR(use_angle_cls=True, lang='fr')


def extract_text_from_image(img_path):
    """./Capture d’écran 2025-07-27 163230 (1).png"""
    results = ocr.ocr(img_path, cls=True)
    text = "\n".join([line[1][0] for line in results[0]]) if results else ""
    return text

def extract_text_from_pdf(pdf_path):
    """Try text extraction first; fallback to OCR if scanned PDF"""
    text_content = ""

    # --- First try native text extraction ---
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + "\n"

    # --- If no text found, fallback to OCR ---
    if not text_content.strip():
        print("⚠️ PDF has no extractable text, running OCR...")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(dpi=300)  # render page as image
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            # Run OCR on the rendered page
            results = ocr.ocr(img, cls=True)
            page_text = "\n".join([line[1][0] for line in results[0]]) if results else ""
            text_content += page_text + "\n"
    else:
        print("✅ Extracted text natively (no OCR needed).")

    return text_content



if __name__ == "__main__":
    # Test image
    img_text = extract_text_from_image("sample1.png")
    print("Image OCR Result:\n", img_text)

    # Test PDF
    # pdf_text = extract_text_from_pdf("sample.pdf")
    # print("PDF Extraction Result:\n", pdf_text[:1000])  # preview first 1000 chars
