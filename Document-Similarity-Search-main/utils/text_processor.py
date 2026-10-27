"""
Text Processor Module
Handles text preprocessing, cleaning, and chunking
Version: 2.0.0 - Enhanced with improved NLTK handling
"""

import re
import nltk
import ssl
from typing import List, Optional
import logging
from collections import Counter

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    MAX_CHUNK_LENGTH,
    STOPWORDS_ENABLED,
    LEMMATIZATION_ENABLED,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_resources() -> None:
    """
    Download required NLTK resources.

    The SSL override is scoped exclusively to this function. It saves the
    original context before patching and restores it in the finally block,
    so no other HTTPS call in the process is affected.
    """
    nltk_resources = {
        'punkt':     'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet':   'corpora/wordnet',
        'omw-1.4':   'corpora/omw-1.4',
    }

    # Scoped SSL override (restored unconditionally in finally)
    original_https_context = getattr(ssl, '_create_default_https_context', None)
    try:
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        for resource_name, resource_path in nltk_resources.items():
            try:
                nltk.data.find(resource_path)
                logger.debug("NLTK resource '%s' already available", resource_name)
            except LookupError:
                # Resource not downloaded at all — standard download
                try:
                    logger.info("Downloading NLTK resource: %s", resource_name)
                    nltk.download(resource_name, quiet=True)
                    logger.info("Downloaded: %s", resource_name)
                except Exception as exc:
                    logger.warning("Could not download %s: %s", resource_name, exc)
            except OSError:
                # Resource directory exists but files are corrupt/incomplete — force re-download
                try:
                    logger.info("Re-downloading corrupt NLTK resource: %s", resource_name)
                    nltk.download(resource_name, quiet=True, force=True)
                    logger.info("Re-downloaded: %s", resource_name)
                except Exception as exc:
                    logger.warning("Could not re-download %s: %s", resource_name, exc)
    finally:
        # Always restore - even if an exception propagates upward
        if original_https_context is not None:
            ssl._create_default_https_context = original_https_context
        elif hasattr(ssl, '_create_default_https_context'):
            del ssl._create_default_https_context


# Initialize NLTK resources
download_nltk_resources()

# Import NLTK components after download
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    logger.error(f"NLTK import error: {e}")
    # Fallback to basic string methods if NLTK fails
    def sent_tokenize(text):
        return text.split('. ')
    
    def word_tokenize(text):
        return text.split()
    
    stopwords = None
    WordNetLemmatizer = None


class TextProcessor:
    """
    Handles all text preprocessing and chunking operations
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize TextProcessor
        
        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize NLTK components with fallback
        try:
            if STOPWORDS_ENABLED and stopwords:
                self.stop_words = set(stopwords.words('english'))
            else:
                self.stop_words = set()
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}")
            self.stop_words = set()
        
        try:
            if LEMMATIZATION_ENABLED and WordNetLemmatizer:
                self.lemmatizer = WordNetLemmatizer()
            else:
                self.lemmatizer = None
        except Exception as e:
            logger.warning(f"Could not initialize lemmatizer: {e}")
            self.lemmatizer = None

    def preprocess(self, text: str, deep_clean: bool = True) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Input text to preprocess
            deep_clean: Whether to perform deep cleaning (URLs, emails, special chars)
            
        Returns:
            Preprocessed and cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Basic cleaning (always applied)
            text = self._basic_clean(text)
            
            if deep_clean:
                # Advanced cleaning operations
                text = self._remove_urls(text)
                text = self._remove_emails(text)
                text = self._remove_special_chars(text)
                text = self._normalize_whitespace(text)
                text = self._remove_extra_punctuation(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return text  # Return original text if preprocessing fails

    def _basic_clean(self, text: str) -> str:
        """
        Perform basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove zero-width characters
        text = text.replace('\u200b', '')
        text = text.replace('\ufeff', '')
        
        return text

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        # Pattern for HTTP/HTTPS URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Pattern for www URLs
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+\.[a-zA-Z]{2,}'
        text = re.sub(www_pattern, '', text)
        
        return text

    def _remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text
        
        Args:
            text: Input text
            
        Returns:
            Text without email addresses
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def _remove_special_chars(self, text: str) -> str:
        """
        Remove special characters while preserving sentence structure
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'\"\(\)]', ' ', text)
        return text

    def _remove_extra_punctuation(self, text: str) -> str:
        """
        Remove excessive punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        # Replace multiple punctuation marks with single
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        return text.strip()

    def chunk_text(self, text: str, method: str = "sentences") -> List[str]:
        """
        Split text into chunks for processing
        
        Args:
            text: Input text to chunk
            method: Chunking method ('sentences', 'tokens', 'paragraphs')
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            if method == "sentences":
                return self._chunk_by_sentences(text)
            elif method == "tokens":
                return self._chunk_by_tokens(text)
            elif method == "paragraphs":
                return self._chunk_by_paragraphs(text)
            else:
                logger.warning(f"Unknown chunking method '{method}', using sentences")
                return self._chunk_by_sentences(text)
                
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            # Fallback to simple splitting
            return self._simple_chunk(text)

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences with overlap
        
        Args:
            text: Input text
            
        Returns:
            List of sentence-based chunks
        """
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if not sentences:
                return [text] if len(text) >= MIN_CHUNK_LENGTH else []
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                # Check if adding sentence exceeds chunk size
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= MIN_CHUNK_LENGTH:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_size = max(1, int(len(current_chunk) * (self.chunk_overlap / self.chunk_size)))
                    current_chunk = current_chunk[-overlap_size:]
                    current_length = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Sentence chunking error: {str(e)}")
            return self._simple_chunk(text)

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """
        Chunk text by token count
        
        Args:
            text: Input text
            
        Returns:
            List of token-based chunks
        """
        try:
            words = text.split()
            
            if not words:
                return []
            
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            
            for i in range(0, len(words), step):
                chunk = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk)
                
                if len(chunk_text) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Token chunking error: {str(e)}")
            return self._simple_chunk(text)

    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs
        
        Args:
            text: Input text
            
        Returns:
            List of paragraph chunks
        """
        try:
            # Split by double newlines or paragraph markers
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Filter and clean paragraphs
            chunks = []
            for para in paragraphs:
                para = para.strip()
                if len(para) >= MIN_CHUNK_LENGTH:
                    chunks.append(para)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            logger.error(f"Paragraph chunking error: {str(e)}")
            return self._simple_chunk(text)

    def _simple_chunk(self, text: str) -> List[str]:
        """
        Simple fallback chunking method based on character count
        
        Args:
            text: Input text
            
        Returns:
            List of chunks
        """
        chunk_size_chars = self.chunk_size * 5  # Approximate 5 chars per token
        overlap_chars = self.chunk_overlap * 5
        
        chunks = []
        
        for i in range(0, len(text), chunk_size_chars - overlap_chars):
            chunk = text[i:i + chunk_size_chars]
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk.strip())
        
        return chunks if chunks else [text]

    def extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """
        Extract top keywords from text based on frequency
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of (word, frequency) tuples
        """
        try:
            # Tokenize
            words = word_tokenize(text.lower())
            
            # Remove stopwords and non-alphabetic tokens
            words = [
                w for w in words 
                if w.isalpha() 
                and len(w) > 2 
                and w not in self.stop_words
            ]
            
            # Lemmatize if enabled
            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(w) for w in words]
            
            # Count frequencies
            word_freq = Counter(words)
            
            return word_freq.most_common(top_n)
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []

    def get_statistics(self, text: str) -> dict:
        """
        Calculate comprehensive text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing text statistics
        """
        try:
            if not text:
                return {}
            
            # Tokenization
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Filter alphabetic words
            alphabetic_words = [w for w in words if w.isalpha()]
            
            # Count unique words
            unique_words = set(w.lower() for w in alphabetic_words)
            
            # Calculate statistics
            stats = {
                "char_count": len(text),
                "word_count": len(alphabetic_words),
                "total_tokens": len(words),
                "sentence_count": max(len(sentences), 1),
                "unique_words": len(unique_words),
                "avg_word_length": (
                    sum(len(w) for w in alphabetic_words) / len(alphabetic_words) 
                    if alphabetic_words else 0
                ),
                "avg_sentence_length": (
                    len(alphabetic_words) / len(sentences) 
                    if sentences else 0
                ),
                "lexical_diversity": (
                    len(unique_words) / len(alphabetic_words) 
                    if alphabetic_words else 0
                ),
                "paragraph_count": len(re.split(r'\n\s*\n', text)),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return {
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            }

    def find_similar_sentences(
        self, 
        text1: str, 
        text2: str, 
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Find similar sentences between two texts using word overlap
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (sentence1, sentence2, similarity) tuples
        """
        try:
            sentences1 = sent_tokenize(text1)
            sentences2 = sent_tokenize(text2)
            
            similar_pairs = []
            
            for s1 in sentences1:
                for s2 in sentences2:
                    # Calculate word overlap similarity
                    words1 = set(word_tokenize(s1.lower()))
                    words2 = set(word_tokenize(s2.lower()))
                    
                    # Remove very short words
                    words1 = {w for w in words1 if len(w) > 2}
                    words2 = {w for w in words2 if len(w) > 2}
                    
                    if words1 and words2:
                        # Jaccard similarity
                        intersection = len(words1 & words2)
                        union = len(words1 | words2)
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity >= threshold:
                            similar_pairs.append((s1, s2, similarity))
            
            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            return similar_pairs
            
        except Exception as e:
            logger.error(f"Similar sentence finding error: {str(e)}")
            return []

    def clean_for_embedding(self, text: str) -> str:
        """
        Clean text specifically for embedding generation
        Removes noise while preserving semantic meaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text optimized for embeddings
        """
        try:
            # Remove URLs and emails (noise for embeddings)
            text = self._remove_urls(text)
            text = self._remove_emails(text)
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            # Remove excessive punctuation
            text = self._remove_extra_punctuation(text)
            
            # Keep sentence structure intact (important for context)
            # Don't lowercase or remove stopwords for embeddings
            
            return text
            
        except Exception as e:
            logger.error(f"Embedding cleaning error: {str(e)}")
            return text

    def validate_text(self, text: str, min_length: int = 50) -> tuple[bool, str]:
        """
        Validate if text is suitable for analysis
        
        Args:
            text: Input text to validate
            min_length: Minimum required text length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text:
            return False, "Text is empty"
        
        if not isinstance(text, str):
            return False, "Text must be a string"
        
        if len(text.strip()) < min_length:
            return False, f"Text too short (minimum {min_length} characters)"
        
        # Check if text contains meaningful content
        words = text.split()
        if len(words) < 10:
            return False, "Text contains too few words"
        
        return True, ""

    def get_reading_time(self, text: str, words_per_minute: int = 200) -> dict:
        """
        Calculate estimated reading time for text
        
        Args:
            text: Input text
            words_per_minute: Average reading speed
            
        Returns:
            Dictionary with reading time information
        """
        try:
            words = word_tokenize(text)
            word_count = len([w for w in words if w.isalpha()])
            
            minutes = word_count / words_per_minute
            
            return {
                "word_count": word_count,
                "estimated_minutes": round(minutes, 1),
                "estimated_seconds": round(minutes * 60),
            }
            
        except Exception as e:
            logger.error(f"Reading time calculation error: {str(e)}")
            return {"word_count": 0, "estimated_minutes": 0, "estimated_seconds": 0}

    def detect_language(self, text: str) -> str:
        """
        Simple language detection (English vs non-English)
        
        Args:
            text: Input text
            
        Returns:
            Detected language code ('en' or 'unknown')
        """
        try:
            # Simple heuristic: check for common English words
            common_english_words = {
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 
                'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 
                'do', 'at', 'this', 'but', 'his', 'by', 'from'
            }
            
            words = set(word_tokenize(text.lower()))
            english_word_count = len(words & common_english_words)
            
            # If more than 5% of unique words are common English words
            if words and english_word_count / len(words) > 0.05:
                return 'en'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return 'unknown'
