"""
Utils Package
Core utilities for plagiarism detection system
"""

from .file_handler import FileHandler
from .text_processor import TextProcessor
from .similarity_engine import SimilarityEngine
from .report_generator import ReportGenerator

__all__ = [
    "FileHandler",
    "TextProcessor",
    "SimilarityEngine",
    "ReportGenerator",
]

__version__ = "2.0.0"
__author__ = "Plagiarism Detection Team"