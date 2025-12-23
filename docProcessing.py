
# docProcessing.py — robust utilities for document loading & chunking

import io
import os
from dataclasses import dataclass, field
from typing import List, Sequence, Optional

# --- MarkItDown & audio backend (FFmpeg is optional via imageio-ffmpeg) ---
from markitdown import MarkItDown

# Provide FFmpeg path for pydub if system FFmpeg is missing
# This prevents "Couldn't find ffmpeg or avconv" warnings from breaking runs.
try:
    from pydub import AudioSegment
    import imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    # If pydub/imageio-ffmpeg aren't installed, MarkItDown still works for most PDFs.
    # Audio/video extraction will be disabled; install pydub + imageio-ffmpeg to enable.
    pass

# --- LangChain splitters: prefer new package, fallback to legacy import ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    # Fallback for environments pinned to LangChain < 0.2
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PDF, OCR, & images ---
try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    raise RuntimeError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

try:
    from PIL import Image
except ModuleNotFoundError:
    raise RuntimeError("Pillow is required. Install with: pip install pillow")

try:
    import pytesseract
except ModuleNotFoundError:
    pytesseract = None  # OCR disabled if Tesseract isn't installed

# Optional: structured markdown extraction from PDFs via pymupdf4llm
try:
    import pymupdf4llm
except ModuleNotFoundError:
    pymupdf4llm = None


# ------------------------------
# Data container for extracted docs
# ------------------------------
@dataclass
class PdfDoc:
    docId: List[str] = field(default_factory=list)
    docSource: List[str] = field(default_factory=list)
    textId: List[int] = field(default_factory=list)
    pageText: List[str] = field(default_factory=list)


# ------------------------------
# Internal helpers
# ------------------------------
def _split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 75) -> List[str]:
    """Split text into chunks using LangChain splitters; fallback to manual."""
    if not text:
        return []
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)
    except Exception:
        return _manual_splitter(text, chunk_size)


def _manual_splitter(text: str, max_size: int) -> List[str]:
    """Simple fixed-size chunker."""
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_size, n)
        chunks.append(text[start:end])
        start = end
    return chunks


# ------------------------------
# Public API
# ------------------------------
def process_files(
    file_paths: Sequence[str],
    doc_number: str,
    doc_name: str,
    chunk_size: int = 525,
    chunk_overlap: int = 100,
) -> PdfDoc:
    """
    Convert arbitrary files (PDF, images, office docs) to text via MarkItDown, then chunk.
    """
    md = MarkItDown()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = PdfDoc()

    for fpath in file_paths:
        result = md.convert(fpath)
        text = getattr(result, "text_content", "") or ""
        chunks = splitter.split_text(text) if text else []
        for i, chunk in enumerate(chunks):
            doc.docId.append(doc_number)
            doc.docSource.append(doc_name)
            doc.textId.append(i)
            doc.pageText.append(chunk)

    return doc


def extract_text_from_image(image: Image.Image) -> str:
    """OCR text from a PIL image using Tesseract."""
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def extract_text_from_pdf(
    pdf_path: str,
    doc_number: str,
    doc_name: str,
    do_split: bool = True,
    chunk_size: int = 525,
    chunk_overlap: int = 100,
) -> PdfDoc:
    """
    Render each PDF page to an image via PyMuPDF, OCR the image, and (optionally) chunk the text.
    """
    source_doc = fitz.open(pdf_path)
    all_text = []

    for i in range(source_doc.page_count):
        page = source_doc[i]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = extract_text_from_image(img)
        if page_text:
            all_text.append(page_text)

    text = "\n".join(all_text)
    doc = PdfDoc()

    chunks = _split_text(text, chunk_size, chunk_overlap) if do_split else [text]
    for idx, chunk in enumerate(chunks):
        doc.docId.append(doc_number)
        doc.docSource.append(doc_name)
        doc.textId.append(idx)
        doc.pageText.append(chunk)

    return doc


def extract_markdown_from_pdf(
    pdf_path: str,
    doc_number: str,
    doc_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 75,
) -> PdfDoc:
    """
    Prefer rich markdown from pymupdf4llm; fallback to OCR if none; then chunk.
    """
    doc = PdfDoc()
    text = ""

    # Structured markdown (if available)
    if pymupdf4llm is not None:
        try:
            text = pymupdf4llm.to_markdown(pdf_path).replace("\ufffd", "")
        except Exception:
            text = ""

    # Fallback OCR if markdown is empty
    if not text:
        text = "\n".join(
            extract_text_from_pdf(pdf_path, doc_number, doc_name, do_split=False).pageText
        )

    chunks = _split_text(text, chunk_size, chunk_overlap)
    for idx, chunk in enumerate(chunks):
        doc.docId.append(doc_number)
        doc.docSource.append(doc_name)
        doc.textId.append(idx)
        doc.pageText.append(chunk)

    return doc


def extract_text_from_xls(xls_path: str, doc_number: str, doc_name: str) -> PdfDoc:
    """
    Extract text from all sheets in an Excel file and return as markdown blocks per sheet.
    """
    import pandas as pd  # import inside to avoid hard dependency if not used

    doc = PdfDoc()
    xls = pd.ExcelFile(xls_path)
    sheets = xls.sheet_names

    for n, sheet in enumerate(sheets):
        df = pd.read_excel(io=xls, sheet_name=sheet, header=None, dtype=str)
        df = df.apply(lambda x: x.str.replace("\n", "<br>") if x.dtype == "object" else x)
        df = df.fillna("")
        md_sheet = f"{sheet}\n{df.to_markdown(index=False)}\n"

        doc.docId.append(doc_number)
        doc.docSource.append(doc_name)
        doc.textId.append(n)
        doc.pageText.append(md_sheet)

    return doc


def crawl_files(
    user_path: str,
    recurse: bool = True,
    allowed_ext: Optional[List[str]] = None,
) -> List[str]:
    """
    Return a list of file paths under user_path. If recurse=True, descend into directories.
    Robust to missing/invalid paths and IO errors; ALWAYS returns a list.
    Optionally filter by file extensions (e.g., ['.pdf', '.png', '.xlsx']).
    """
    file_list: List[str] = []

    if not user_path:
        return file_list

    abs_path = os.path.abspath(user_path)
    if not os.path.isdir(abs_path):
        return file_list

    # Normalize extension filter
    if allowed_ext is not None:
        allowed_ext = [e.lower() for e in allowed_ext]

    try:
        with os.scandir(abs_path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        if recurse:
                            nested = crawl_files(entry.path, recurse=True, allowed_ext=allowed_ext)
                            if isinstance(nested, list):
                                file_list.extend(nested)
                    else:
                        if allowed_ext is None:
                            file_list.append(entry.path)
                        else:
                            _, ext = os.path.splitext(entry.name)
                            if ext.lower() in allowed_ext:
                                file_list.append(entry.path)
                except Exception:
                    # Skip problematic entries but keep going
                    continue
    except Exception:
        # scandir failed; return what we have (empty list)
        return file_list

    return file_list


def load_content_path(config_path: str) -> str:
    """
    Load contentPath from manConfig.json and resolve it relative to the config file's directory.
    Returns an absolute path.
    """
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw = cfg.get("data", {}).get("contentPath", "./data")
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.abspath(os.path.join(base, raw))


def to_dataframe(pdf_doc: PdfDoc):
    """
    Convert PdfDoc to a pandas DataFrame: columns [doc_id, doc_source, text_id, text]
    """
    import pandas as pd
    return pd.DataFrame(
        {
            "doc_id": pdf_doc.docId,
            "doc_source": pdf_doc.docSource,
            "text_id": pdf_doc.textId,
            "text": pdf_doc.pageText,
        }
    )


# ------------------------------
# Backward-compatibility aliases (camelCase)
# ------------------------------
def processFile(file_paths: Sequence[str], doc_number: str, doc_name: str,
                chunk_size: int = 525, chunk_overlap: int = 100) -> PdfDoc:
    return process_files(file_paths, doc_number, doc_name, chunk_size, chunk_overlap)


def extractTextFromImage(image: Image.Image) -> str:
    return extract_text_from_image(image)


def extractTextFromPdf(pdf_path: str, doc_number: str, doc_name: str,
                       do_split: bool = True, chunk_size: int = 525, chunk_overlap: int = 100) -> PdfDoc:
    return extract_text_from_pdf(pdf_path, doc_number, doc_name, do_split, chunk_size, chunk_overlap)


def extractMarkdownFromPdf(pdf_path: str, doc_number: str, doc_name: str,
                           chunk_size: int = 800, chunk_overlap: int = 75) -> PdfDoc:
    return extract_markdown_from_pdf(pdf_path, doc_number, doc_name, chunk_size, chunk_overlap)


def extractTextFromXls(xls_path: str, doc_number: str, doc_name: str) -> PdfDoc:
    return extract_text_from_xls(xls_path, doc_number, doc_name)


def crawlFiles(user_path: str, recurse: bool = True, allowed_ext: Optional[List[str]] = None) -> List[str]:
    # Properly completed alias body
    return crawl_files(user_path, recurse, allowed_ext)
    

