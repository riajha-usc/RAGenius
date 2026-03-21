import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP, SAMPLE_DOCS_DIR


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?$()\-/%\'\"@#&*+=$]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def normalize_financial(text: str) -> str:
        text = re.sub(r'\$\s+', '$', text)
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        text = re.sub(r'(?i)(q[1-4])\s+(\d{4})', r'\1 \2', text)
        text = re.sub(r'(?i)(fy)\s*(\d{2,4})', r'FY\2', text)
        return text


class DocumentChunker:

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "chunk_index": len(chunks),
                }
                if metadata:
                    chunk_data.update(metadata)
                chunks.append(chunk_data)

                overlap_chars = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    overlap_chars += len(s)
                    overlap_sentences.insert(0, s)
                    if overlap_chars >= self.chunk_overlap:
                        break
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                "text": chunk_text,
                "char_count": len(chunk_text),
                "chunk_index": len(chunks),
            }
            if metadata:
                chunk_data.update(metadata)
            chunks.append(chunk_data)

        return chunks


class DocumentLoader:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()

    def load_pdf(self, filepath: str) -> str:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def load_text(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def load_document(self, filepath: str) -> str:
        path = Path(filepath)
        if path.suffix.lower() == ".pdf":
            return self.load_pdf(filepath)
        return self.load_text(filepath)

    def process_document(self, filepath: str) -> List[Dict]:
        path = Path(filepath)
        raw_text = self.load_document(filepath)
        cleaned = self.cleaner.clean(raw_text)
        normalized = self.cleaner.normalize_financial(cleaned)

        metadata = {
            "source": path.name,
            "file_type": path.suffix.lower(),
            "total_chars": len(normalized),
        }

        return self.chunker.chunk_text(normalized, metadata)

    def process_directory(self, directory: str = None) -> pd.DataFrame:
        dir_path = Path(directory) if directory else SAMPLE_DOCS_DIR
        all_chunks = []

        for filepath in dir_path.iterdir():
            if filepath.suffix.lower() in [".pdf", ".txt", ".md", ".csv"]:
                try:
                    chunks = self.process_document(str(filepath))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {filepath.name}: {e}")

        if not all_chunks:
            return pd.DataFrame()

        df = pd.DataFrame(all_chunks)
        df["doc_id"] = df.groupby("source").ngroup()
        df["chunk_id"] = range(len(df))
        return df


class DataPipeline:

    def __init__(self):
        self.loader = DocumentLoader()

    def run(self, directory: str = None) -> pd.DataFrame:
        df = self.loader.process_directory(directory)
        if df.empty:
            return df

        df["text_length"] = df["text"].apply(len)
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))

        stats = self._compute_stats(df)
        return df

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        return {
            "total_documents": df["source"].nunique(),
            "total_chunks": len(df),
            "avg_chunk_length": np.mean(df["text_length"]),
            "std_chunk_length": np.std(df["text_length"]),
            "median_chunk_length": np.median(df["text_length"]),
            "avg_word_count": np.mean(df["word_count"]),
            "total_words": np.sum(df["word_count"]),
            "documents": df.groupby("source").agg(
                chunks=("chunk_id", "count"),
                avg_length=("text_length", "mean"),
            ).to_dict("index"),
        }

    def process_single_text(self, text: str, source: str = "direct_input") -> pd.DataFrame:
        cleaner = TextCleaner()
        chunker = DocumentChunker()
        cleaned = cleaner.clean(text)
        normalized = cleaner.normalize_financial(cleaned)
        chunks = chunker.chunk_text(normalized, {"source": source})
        df = pd.DataFrame(chunks)
        df["doc_id"] = 0
        df["chunk_id"] = range(len(df))
        df["text_length"] = df["text"].apply(len)
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        return df
