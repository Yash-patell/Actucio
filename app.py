import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
from decouple import config
import requests
import json
import logging
import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
import io

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the path to the Tesseract executable, agar aapke system me install hai
# Windows me default path "C:\Program Files\Tesseract-OCR\tesseract.exe" hota hai.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load API key from .env file
OPENROUTER_API_KEY = config('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI()

# Enable CORS for frontend communication
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Actucio backend API!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    try:
        # File ka content read karein
        file_content = await file.read()
        
        # File type ke hisab se text extract karein
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            text = extract_text_from_image(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from the document. The document might be a low-quality scan or encrypted.")

        # Extracted text ko LLM ko bhejein
        extracted_data = await extract_data_with_llm(text)

        return {"filename": file.filename, "extracted_data": extracted_data}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error during file upload or processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return None

async def extract_data_with_llm(document_text: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    model_name = "mistralai/mistral-7b-instruct"

    # Yahan hum LLM ko specific instructions denge
    prompt = f"""
    You are an expert at extracting financial and legal information from company documents.
    Extract the following data fields from the text below. If a field is not found, use a null value.
    Ensure the output is a valid JSON object.

    Data Fields:
    1.  company_name
    2.  ceo_name
    3.  turnover
    4.  profit

    Document Text:
    ---
    {document_text}
    ---
    """

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    try:
        logging.info("Calling OpenRouter API for data extraction...")
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        result = response.json()
        extracted_content = result['choices'][0]['message']['content']
        logging.info("API call successful. Extracted content: %s", extracted_content)

        return json.loads(extracted_content)

    except requests.exceptions.RequestException as e:
        logging.error(f"OpenRouter API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM API Error: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from LLM response: {e}")
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except KeyError as e:
        logging.error(f"Unexpected API response format: {e}")
        raise HTTPException(status_code=500, detail="Unexpected API response format.")