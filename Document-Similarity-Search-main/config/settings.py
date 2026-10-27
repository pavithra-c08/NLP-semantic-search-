"""
Configuration Settings
Centralized configuration for the plagiarism detection system
"""

import os
from pathlib import Path

# Application Metadata
APP_TITLE = "Document Similarity Search"
APP_ICON = "📄"
VERSION = "2.1.0"

# File Processing
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "csv", "xlsx"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES = 500  # Maximum pages to process

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = Path.home() / ".cache" / "sentence-transformers"
EMBEDDING_DIMENSION = 384
# all-MiniLM-L6-v2 silently truncates beyond 256 word-pieces.
# We truncate full-doc text to this many characters before embedding
# to avoid invisible information loss on long documents.
MAX_EMBEDDING_CHARS = 8000  # ~1800 words, well within the 256-token limit

# Text Processing
CHUNK_SIZE = 500  # Default chunk size in tokens
CHUNK_OVERLAP = 50  # Overlap between chunks
MIN_CHUNK_LENGTH = 50  # Minimum characters per chunk
MAX_CHUNK_LENGTH = 2000  # Maximum characters per chunk

# Similarity Thresholds
SIMILARITY_THRESHOLDS = {
    "high": 0.80,  # High plagiarism risk
    "moderate": 0.50,  # Moderate similarity
    "low": 0.30,  # Low similarity
}

# Analysis Modes
# Each mode governs chunking strategy, heatmap resolution, and match depth.
ANALYSIS_MODES = {
    "Quick": {
        "chunk_size": 1000,
        "overlap": 25,
        "chunk_method": "paragraphs",
        "top_matches": 5,
        "heatmap_limit": 20,
    },
    "Standard": {
        "chunk_size": 500,
        "overlap": 50,
        "chunk_method": "sentences",
        "top_matches": 10,
        "heatmap_limit": 35,
    },
    "Deep": {
        "chunk_size": 250,
        "overlap": 100,
        "chunk_method": "tokens",
        "top_matches": 20,
        "heatmap_limit": 50,
    },
}

# Language Processing
STOPWORDS_ENABLED = True
LEMMATIZATION_ENABLED = True

# Report Generation
REPORT_TITLE = "Plagiarism Analysis Report"
REPORT_AUTHOR = "AI Detection System"
EXPORT_JSON_INDENT = 2

# Performance Settings
BATCH_SIZE = 32  # Batch size for embeddings
USE_GPU = True   # Uses GPU if available, falls back to CPU automatically
CACHE_EMBEDDINGS = True

# UI Configuration
SHOW_PROGRESS_BAR = True
SHOW_DEBUG_INFO = False

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "plagiarism_checker.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = 300  # seconds for long operations

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMP_DIR = PROJECT_ROOT / "temp"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Create directories if they don't exist
for directory in [TEMP_DIR, EXPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Environment-specific overrides
ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    SHOW_DEBUG_INFO = False
    LOG_LEVEL = "WARNING"
    CACHE_EMBEDDINGS = True
    USE_GPU = True
elif ENV == "development":
    SHOW_DEBUG_INFO = True
    LOG_LEVEL = "DEBUG"
elif ENV == "testing":
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for testing
    BATCH_SIZE = 16

# Feature Flags (implemented features only)
FEATURES = {
    "heatmap_visualization": True,
    "detailed_statistics": True,
    "matching_highlights": True,
    "export_pdf": True,
    "export_excel": True,
    "export_json": True,
    "analysis_history": True,
}

# Constants
EPSILON = 1e-8  # Small value to prevent division by zero
MIN_TEXT_LENGTH = 100  # Minimum text length to analyze
MAX_TEXT_LENGTH = 10_000_000  # Maximum text length (10M chars)

# Validation Rules
VALIDATION_RULES = {
    "min_file_size": 100,  # bytes
    "max_file_size": MAX_FILE_SIZE,
    "allowed_extensions": SUPPORTED_FORMATS,
    "forbidden_patterns": [r"\.exe$", r"\.dll$", r"\.bat$"],
}

# User Messages
MESSAGES = {
    "file_too_large": f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)}MB",
    "file_too_small": "File is too small to analyze",
    "invalid_format": f"Invalid file format. Supported: {', '.join(SUPPORTED_FORMATS)}",
    "extraction_failed": "Failed to extract text from document",
    "analysis_failed": "Analysis failed. Please try again.",
    "analysis_success": "Analysis completed successfully!",
}

# Help Text
HELP_TEXT = {
    "similarity_threshold": "Minimum cosine score to flag a chunk pair as a match",
    "chunk_size": "Size of text segments. Smaller = more granular but slower",
    "analysis_mode": "Quick: paragraphs, fast | Standard: sentences, balanced | Deep: tokens, thorough",
    "export_format": "Choose format for downloading analysis results",
}
