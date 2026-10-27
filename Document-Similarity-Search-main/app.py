"""
Advanced Document Similarity & Plagiarism Detection System
Main Streamlit Application
Version: 2.1.0
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import (
    APP_TITLE,
    APP_ICON,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE,
    SIMILARITY_THRESHOLDS,
    ANALYSIS_MODES,
    MODEL_NAME,
)
from utils.file_handler import FileHandler
from utils.text_processor import TextProcessor
from utils.similarity_engine import SimilarityEngine
from utils.report_generator import ReportGenerator


# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
def load_custom_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# Pre-flight check
def _preflight_check() -> None:
    """
    Validate that all required resources are available before accepting uploads.
    Surfaces clear, actionable errors rather than mysterious mid-analysis failures.
    """
    errors = []
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        errors.append(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    from config.settings import TEMP_DIR, EXPORTS_DIR
    for d in [TEMP_DIR, EXPORTS_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Cannot create required directory {d}: {exc}")
    if errors:
        for msg in errors:
            st.error(f"⚠️ Pre-flight check failed: {msg}")
        st.stop()


_preflight_check()


# Initialize Session State
def init_session_state():
    """Initialize session state variables"""
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "history" not in st.session_state:
        st.session_state.history = []


init_session_state()


# Sidebar Configuration
def render_sidebar():
    """Render sidebar with advanced options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Analysis Settings")
        analysis_mode = st.selectbox(
            "Analysis Depth",
            list(ANALYSIS_MODES.keys()),
            index=1,
            help="Quick: paragraph chunks, fastest | Standard: sentence chunks, balanced | Deep: token chunks, most granular",
        )
        mode_cfg = ANALYSIS_MODES[analysis_mode]
        st.caption(
            f"Chunk method: **{mode_cfg['chunk_method']}** | "
            f"Chunk size: **{mode_cfg['chunk_size']}** tokens | "
            f"Top matches: **{mode_cfg['top_matches']}**"
        )

        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum cosine score to flag a chunk pair as a match",
        )
        
        st.divider()
        
        st.subheader("Advanced Options")
        show_heatmap = st.checkbox("Show Similarity Heatmap", value=True)
        show_highlights = st.checkbox("Highlight Matching Segments", value=True)
        show_statistics = st.checkbox("Show Detailed Statistics", value=True)
        
        st.divider()
        
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Report Format",
            ["PDF", "Excel", "JSON"],
            help="Choose export format for results",
        )
        
        st.divider()
        
        # System Info
        with st.expander("📊 System Information"):
            st.caption(f"**Version**: 2.1.0")
            st.caption(f"**Model**: {MODEL_NAME}")
            st.caption(f"**Max File Size**: {MAX_FILE_SIZE // (1024*1024)}MB")
            st.caption(f"**Supported Formats**: {', '.join(SUPPORTED_FORMATS)}")
        
        return {
            "analysis_mode": analysis_mode,
            "similarity_threshold": similarity_threshold,
            "show_heatmap": show_heatmap,
            "show_highlights": show_highlights,
            "show_statistics": show_statistics,
            "export_format": export_format,
        }


# Main Header
def render_header():
    """Render main application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown(
            """
            <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
                Enterprise-grade plagiarism detection with advanced semantic analysis
            </div>
            """,
            unsafe_allow_html=True,
        )


# File Upload Section
def render_upload_section():
    """Render file upload interface"""
    st.subheader("📁 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader(
            "Document 1 (Source)",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            key="file1",
        )
        if file1:
            st.success(f"✓ {file1.name} ({file1.size / 1024:.1f} KB)")
    
    with col2:
        file2 = st.file_uploader(
            "Document 2 (Comparison)",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            key="file2",
        )
        if file2:
            st.success(f"✓ {file2.name} ({file2.size / 1024:.1f} KB)")
    
    return file1, file2


# Analysis Section
def perform_analysis(file1, file2, config):
    """Perform plagiarism analysis."""

    # analysis_mode drives chunk_size, overlap, and chunk_method
    mode_cfg = ANALYSIS_MODES[config["analysis_mode"]]
    file_handler = FileHandler()
    text_processor = TextProcessor(
        chunk_size=mode_cfg["chunk_size"],
        chunk_overlap=mode_cfg["overlap"],
    )
    similarity_engine = SimilarityEngine()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📄 Extracting text from documents...")
        progress_bar.progress(10)
        text1 = file_handler.extract_text(file1)
        text2 = file_handler.extract_text(file2)

        if not text1 or not text2:
            st.error(
                "❌ Could not extract text from one or both documents. "
                "Check the file is not password-protected or corrupted."
            )
            return None

        progress_bar.progress(25)

        status_text.text("🔧 Preprocessing and cleaning text...")
        processed1 = text_processor.preprocess(text1)
        processed2 = text_processor.preprocess(text2)
        progress_bar.progress(40)

        # chunk_method is driven by the selected analysis_mode
        chunk_method = mode_cfg["chunk_method"]
        status_text.text(f"✂️ Chunking documents ({chunk_method})...")
        chunks1 = text_processor.chunk_text(processed1, method=chunk_method)
        chunks2 = text_processor.chunk_text(processed2, method=chunk_method)

        if not chunks1 or not chunks2:
            st.error("❌ Documents produced no text chunks. They may be too short.")
            return None

        progress_bar.progress(55)

        status_text.text("🧠 Computing semantic similarity...")
        results = similarity_engine.analyze(
            text1=text1,
            text2=text2,
            chunks1=chunks1,
            chunks2=chunks2,
            threshold=config["similarity_threshold"],
            analysis_mode=config["analysis_mode"],
        )
        progress_bar.progress(85)

        status_text.text("📊 Generating insights...")
        results["file1_name"] = file1.name
        results["file2_name"] = file2.name
        results["config"] = config
        progress_bar.progress(100)

        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        return results

    except RuntimeError as exc:
        # Raised explicitly by SimilarityEngine when model inference fails
        st.error(f"❌ Model error: {exc}")
        progress_bar.empty()
        status_text.empty()
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error during analysis: {exc}")
        progress_bar.empty()
        status_text.empty()
        return None


# Results Display Section
def display_results(results, config):
    """Display analysis results"""
    
    st.divider()
    st.header("📊 Analysis Results")

    # ── Plagiarism Score gauge ────────────────────────────────────────────────
    p_score = results["plagiarism_score"]
    if p_score >= 80:
        bar_color = "#ef4444"   # red
        badge_color = "#fef2f2"
        badge_text_color = "#991b1b"
        label = "🔴 High Risk"
    elif p_score >= 50:
        bar_color = "#f59e0b"   # amber
        badge_color = "#fffbeb"
        badge_text_color = "#92400e"
        label = "🟡 Moderate Risk"
    elif p_score >= 30:
        bar_color = "#f97316"   # orange
        badge_color = "#fff7ed"
        badge_text_color = "#9a3412"
        label = "🟠 Low-Moderate Risk"
    else:
        bar_color = "#22c55e"   # green
        badge_color = "#f0fdf4"
        badge_text_color = "#166534"
        label = "🟢 Low Risk"

    st.markdown(
        f"""
        <div style="background:{'#1e1e1e' if True else '#f9fafb'};border-radius:12px;
                    padding:1.5rem 2rem;margin-bottom:1rem;
                    border:1px solid #333;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
            <span style="font-size:1rem;font-weight:600;color:#e5e7eb;">Plagiarism Score</span>
            <span style="background:{badge_color};color:{badge_text_color};
                         padding:0.25rem 0.75rem;border-radius:999px;
                         font-size:0.85rem;font-weight:600;">{label}</span>
          </div>
          <div style="font-size:2.8rem;font-weight:700;color:{bar_color};
                      line-height:1;margin-bottom:0.75rem;">{p_score:.1f}%</div>
          <div style="background:#374151;border-radius:999px;height:10px;width:100%;">
            <div style="background:{bar_color};border-radius:999px;
                        height:10px;width:{min(p_score,100):.1f}%;
                        transition:width 0.4s ease;"></div>
          </div>
          <div style="font-size:0.78rem;color:#9ca3af;margin-top:0.5rem;">
            Based on chunk-level content matching — how much of Document 1 appears in Document 2
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary metric cards ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        similarity_score = results["overall_similarity"]
        st.metric(
            "Semantic Similarity",
            f"{similarity_score:.1%}",
            help="Full-document cosine similarity — measures topic closeness, not copying.",
        )

    with col2:
        risk_level = results["risk_level"]
        risk_icons = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
        st.metric("Risk Level", f"{risk_icons.get(risk_level, '')} {risk_level}")

    with col3:
        st.metric("Matching Segments", f"{results['matching_segments']}")

    with col4:
        st.metric(
            "Doc1 Unique Chunks",
            f"{results['unique_percentage']:.1%}",
            help="Percentage of Document 1 chunks with no similar counterpart in Document 2.",
        )

    st.divider()

    # ── Detailed Interpretation ───────────────────────────────────────────────
    st.subheader("🔍 Detailed Analysis")

    interpretation = results.get("interpretation", "")
    if p_score >= 80:
        st.error(f"⚠️ **High Plagiarism Risk Detected**\n\n{interpretation}")
    elif p_score >= 50:
        st.warning(f"⚠️ **Moderate Plagiarism Risk**\n\n{interpretation}")
    else:
        st.success(f"✅ **Low Plagiarism Risk — Mostly Original**\n\n{interpretation}")
    
    # Heatmap
    if config["show_heatmap"] and "heatmap_data" in results:
        st.subheader("🗺️ Similarity Heatmap")
        st.plotly_chart(results["heatmap_data"], use_container_width=True)
    
    # Statistics
    if config["show_statistics"]:
        st.subheader("📈 Statistical Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document 1 Statistics**")
            stats1 = results.get("doc1_stats", {})
            st.write(f"- Word Count: {stats1.get('word_count', 0):,}")
            st.write(f"- Sentence Count: {stats1.get('sentence_count', 0):,}")
            st.write(f"- Unique Words: {stats1.get('unique_words', 0):,}")
        
        with col2:
            st.markdown("**Document 2 Statistics**")
            stats2 = results.get("doc2_stats", {})
            st.write(f"- Word Count: {stats2.get('word_count', 0):,}")
            st.write(f"- Sentence Count: {stats2.get('sentence_count', 0):,}")
            st.write(f"- Unique Words: {stats2.get('unique_words', 0):,}")
    
    # Matching Segments
    if config["show_highlights"] and "matching_segments_detail" in results:
        st.subheader("🎯 Matching Text Segments")
        
        for idx, segment in enumerate(results["matching_segments_detail"][:5], 1):
            with st.expander(f"Match {idx} - Similarity: {segment['score']:.1%}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Document 1:**\n{segment['text1']}")
                with col2:
                    st.markdown(f"**Document 2:**\n{segment['text2']}")


# Export Section
def render_export_section(results, config):
    """Render export options"""
    
    st.divider()
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    report_gen = ReportGenerator()
    
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_data = report_gen.generate_pdf(results)
                if pdf_data:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_data,
                        file_name="plagiarism_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
    
    with col2:
        if st.button("📊 Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                excel_data = report_gen.generate_excel(results)
                if excel_data:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=excel_data,
                        file_name="plagiarism_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
    
    with col3:
        if st.button("📋 Copy JSON Data", use_container_width=True):
            json_data = report_gen.generate_json(results)
            st.code(json_data, language="json")


# Main Application Flow
def main():
    """Main application entry point"""
    
    # Render components
    render_header()
    config = render_sidebar()
    file1, file2 = render_upload_section()
    
    st.divider()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Documents",
            type="primary",
            use_container_width=True,
            disabled=not (file1 and file2),
        )
    
    # Perform analysis
    if analyze_button and file1 and file2:
        # Validate file sizes
        if file1.size > MAX_FILE_SIZE or file2.size > MAX_FILE_SIZE:
            st.error(f"❌ File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
            return

        # Clear previous results so stale data never shows under a new analysis
        st.session_state.analysis_complete = False
        st.session_state.results = None

        results = perform_analysis(file1, file2, config)
        
        if results:
            st.session_state.results = results
            st.session_state.analysis_complete = True
            
            # Add to history
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file1": file1.name,
                "file2": file2.name,
                "similarity": results["overall_similarity"],
                "mode": config["analysis_mode"],
            })
    
    # Display results if available
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, config)
        render_export_section(st.session_state.results, config)
    
    # History section
    if st.session_state.history:
        with st.sidebar:
            st.divider()
            with st.expander("📜 Analysis History"):
                for item in reversed(st.session_state.history[-5:]):
                    st.caption(
                        f"{item['timestamp']}: {item['file1']} vs {item['file2']} "
                        f"({item['similarity']:.1%}, {item.get('mode', 'Standard')})"
                    )


# Footer
def render_footer():
    """Render application footer"""
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>Advanced Plagiarism Detection System v2.1.0</p>
            <p>Powered by Sentence Transformers & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    render_footer()