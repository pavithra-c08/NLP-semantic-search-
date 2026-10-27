"""
Report Generator Module
Handles export of analysis results in various formats
"""

import json
import io
from datetime import datetime
from typing import Dict, Optional
import logging
import pandas as pd
from fpdf import FPDF

from config.settings import (
    REPORT_TITLE,
    EXPORT_JSON_INDENT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates reports in multiple formats (PDF, Excel, JSON)
    """

    def __init__(self):
        """Initialize ReportGenerator"""
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_pdf(self, results: Dict) -> Optional[bytes]:
        """
        Generate PDF report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            PDF file as bytes
        """
        try:
            pdf = PDFReport()
            
            # Add title page
            pdf.add_page()
            pdf.chapter_title("Plagiarism Analysis Report")
            pdf.add_metadata(self.timestamp, results)
            
            # Add summary
            pdf.add_page()
            pdf.chapter_title("Executive Summary")
            pdf.add_summary(results)
            
            # Add detailed analysis
            pdf.add_page()
            pdf.chapter_title("Detailed Analysis")
            pdf.add_details(results)
            
            # Add matching segments
            if results.get("matching_segments_detail"):
                pdf.add_page()
                pdf.chapter_title("Matching Segments")
                pdf.add_matching_segments(results["matching_segments_detail"][:5])
            
            # Add statistics
            pdf.add_page()
            pdf.chapter_title("Document Statistics")
            pdf.add_statistics(results.get("doc1_stats", {}), results.get("doc2_stats", {}))
            
            # Output to bytes
            pdf_output = pdf.output(dest='S')
            
            # Convert to bytes if it's a string
            if isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin-1')
            else:
                pdf_bytes = bytes(pdf_output)
            
            logger.info("PDF report generated successfully")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            return None

    def generate_excel(self, results: Dict) -> Optional[bytes]:
        """
        Generate Excel report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Excel file as bytes
        """
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    "Metric": [
                        "Overall Similarity",
                        "Plagiarism Score",
                        "Risk Level",
                        "Matching Segments",
                        "Doc1 Unique Chunks (no match in Doc2)",
                        "Analysis Date",
                        "Analysis Mode",
                    ],
                    "Value": [
                        f"{results['overall_similarity']:.2%}",
                        f"{results['plagiarism_score']:.2f}%",
                        results['risk_level'],
                        results['matching_segments'],
                        f"{results['unique_percentage']:.2%}",  # fraction of doc1 chunks with no match
                        self.timestamp,
                        results.get('analysis_mode', 'Standard'),
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Document 1 statistics
                if results.get('doc1_stats'):
                    stats1 = results['doc1_stats']
                    df_stats1 = pd.DataFrame([stats1])
                    df_stats1.to_excel(writer, sheet_name='Document 1 Stats', index=False)
                
                # Document 2 statistics
                if results.get('doc2_stats'):
                    stats2 = results['doc2_stats']
                    df_stats2 = pd.DataFrame([stats2])
                    df_stats2.to_excel(writer, sheet_name='Document 2 Stats', index=False)
                
                # Matching segments
                if results.get('matching_segments_detail'):
                    segments_data = []
                    for seg in results['matching_segments_detail'][:20]:
                        segments_data.append({
                            'Similarity': f"{seg['score']:.2%}",
                            'Position Doc1': seg['position1'],
                            'Position Doc2': seg['position2'],
                            'Text Doc1': seg['text1'][:200],
                            'Text Doc2': seg['text2'][:200],
                        })
                    df_segments = pd.DataFrame(segments_data)
                    df_segments.to_excel(writer, sheet_name='Matching Segments', index=False)
            
            excel_bytes = output.getvalue()
            logger.info("Excel report generated successfully")
            return excel_bytes
            
        except Exception as e:
            logger.error(f"Excel generation error: {str(e)}")
            return None

    def generate_json(self, results: Dict) -> str:
        """
        Generate JSON report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            JSON string
        """
        try:
            # Create a clean copy without non-serializable objects
            clean_results = {
                "metadata": {
                    "analysis_date": self.timestamp,
                    "report_title": REPORT_TITLE,
                },
                "summary": {
                    "overall_similarity": results.get("overall_similarity", 0),
                    "plagiarism_score": results.get("plagiarism_score", 0),
                    "risk_level": results.get("risk_level", "Unknown"),
                    "matching_segments": results.get("matching_segments", 0),
                    "unique_percentage": results.get("unique_percentage", 0),
                },
                "details": {
                    "file1_name": results.get("file1_name", "Document 1"),
                    "file2_name": results.get("file2_name", "Document 2"),
                    "analysis_mode": results.get("analysis_mode", "Standard"),
                    "threshold_used": results.get("threshold_used", 0.7),
                },
                "statistics": {
                    "document1": results.get("doc1_stats", {}),
                    "document2": results.get("doc2_stats", {}),
                },
                "interpretation": results.get("interpretation", ""),
            }
            
            # Add matching segments (limited)
            if results.get("matching_segments_detail"):
                clean_results["matching_segments"] = [
                    {
                        "similarity": seg["score"],
                        "position1": seg["position1"],
                        "position2": seg["position2"],
                        "text1_preview": seg["text1"][:100],
                        "text2_preview": seg["text2"][:100],
                    }
                    for seg in results["matching_segments_detail"][:10]
                ]
            
            json_str = json.dumps(clean_results, indent=EXPORT_JSON_INDENT)
            logger.info("JSON report generated successfully")
            return json_str
            
        except Exception as e:
            logger.error(f"JSON generation error: {str(e)}")
            return json.dumps({"error": str(e)})


class PDFReport(FPDF):
    """
    Custom PDF report class
    """

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """PDF header"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Plagiarism Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """PDF footer"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title: str):
        """Add chapter title"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body: str):
        """Add chapter body"""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

    def add_metadata(self, timestamp: str, results: Dict):
        """Add report metadata"""
        self.set_font('Arial', '', 11)
        self.cell(0, 7, f"Generated: {timestamp}", 0, 1)
        self.cell(0, 7, f"Document 1: {results.get('file1_name', 'N/A')}", 0, 1)
        self.cell(0, 7, f"Document 2: {results.get('file2_name', 'N/A')}", 0, 1)
        self.ln(5)

    def add_summary(self, results: Dict):
        """Add executive summary"""
        self.set_font('Arial', 'B', 11)
        self.cell(0, 7, f"Plagiarism Score: {results['plagiarism_score']:.1f}%  (chunk-based)", 0, 1)
        self.cell(0, 7, f"Semantic Similarity: {results['overall_similarity']:.2%}  (full-document)", 0, 1)
        self.cell(0, 7, f"Risk Level: {results['risk_level']}", 0, 1)
        self.cell(0, 7, f"Matching Segments: {results['matching_segments']}", 0, 1)
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, results.get('interpretation', ''))
        self.ln()

    def add_details(self, results: Dict):
        """Add detailed analysis"""
        self.set_font('Arial', '', 11)
        details = f"""
Analysis Mode: {results.get('analysis_mode', 'Standard')}
Threshold Used: {results.get('threshold_used', 0.7):.2f}
Unique Content: {results.get('unique_percentage', 0):.2%}

This analysis uses advanced natural language processing and semantic similarity 
algorithms to detect potential plagiarism. The results should be used as a guide 
and verified through manual review.
        """
        self.multi_cell(0, 7, details.strip())
        self.ln()

    def add_matching_segments(self, segments: list):
        """Add matching segments"""
        self.set_font('Arial', '', 10)
        
        for idx, seg in enumerate(segments, 1):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 7, f"Match {idx} (Similarity: {seg['score']:.2%})", 0, 1)
            
            self.set_font('Arial', '', 9)
            self.multi_cell(0, 5, f"Document 1: {seg['text1'][:200]}...")
            self.multi_cell(0, 5, f"Document 2: {seg['text2'][:200]}...")
            self.ln(3)

    def add_statistics(self, stats1: Dict, stats2: Dict):
        """Add document statistics"""
        self.set_font('Arial', 'B', 11)
        self.cell(0, 7, "Document 1 Statistics:", 0, 1)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"Word Count: {stats1.get('word_count', 0):,}", 0, 1)
        self.cell(0, 6, f"Sentence Count: {stats1.get('sentence_count', 0):,}", 0, 1)
        self.cell(0, 6, f"Unique Words: {stats1.get('unique_words', 0):,}", 0, 1)
        self.ln(5)
        
        self.set_font('Arial', 'B', 11)
        self.cell(0, 7, "Document 2 Statistics:", 0, 1)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"Word Count: {stats2.get('word_count', 0):,}", 0, 1)
        self.cell(0, 6, f"Sentence Count: {stats2.get('sentence_count', 0):,}", 0, 1)
        self.cell(0, 6, f"Unique Words: {stats2.get('unique_words', 0):,}", 0, 1)