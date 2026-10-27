"""
Similarity Engine Module
Core semantic similarity and plagiarism detection logic
Version: 2.1.0

Fixes applied:
  - Model loaded via module-level @st.cache_resource (not instance method)
  - Embeddings computed once per analysis; heatmap reuses them (no duplicate inference)
  - Chunk matching uses NumPy masking instead of O(n*m) Python loop
  - analysis_mode now governs chunk_method, heatmap_limit, and top_matches
  - unique_percentage is chunk-based (fraction of doc1 with no match) - semantically correct
  - Embedding failures raise RuntimeError instead of silently returning a zero vector
"""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

from config.settings import (
    MODEL_NAME,
    SIMILARITY_THRESHOLDS,
    BATCH_SIZE,
    ANALYSIS_MODES,
    MAX_EMBEDDING_CHARS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level cached model loader
# Decorated at module scope so the cache key is stable regardless of how many
# SimilarityEngine instances are created.  All instances share the same model.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading semantic model...")
def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (runs once per process)."""
    logger.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully")
    return model


class SimilarityEngine:
    """
    Handles semantic similarity computation and analysis
    """

    def __init__(self):
        """Initialize SimilarityEngine with shared cached model."""
        self.model: SentenceTransformer = _get_model()
        self.thresholds = SIMILARITY_THRESHOLDS

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Return a normalised embedding for text.

        Truncates to MAX_EMBEDDING_CHARS before encoding because
        all-MiniLM-L6-v2 silently drops tokens beyond its 256-piece limit,
        which would cause invisible information loss on long documents.

        Raises RuntimeError instead of silently returning a zero vector.
        """
        try:
            truncated = text[:MAX_EMBEDDING_CHARS] if len(text) > MAX_EMBEDDING_CHARS else text
            return self.model.encode(
                [truncated],
                show_progress_bar=False,
                normalize_embeddings=True,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Return normalised embeddings for a list of texts."""
        try:
            truncated = [
                t[:MAX_EMBEDDING_CHARS] if len(t) > MAX_EMBEDDING_CHARS else t
                for t in texts
            ]
            return self.model.encode(
                truncated,
                batch_size=BATCH_SIZE,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Batch embedding failed: {exc}") from exc

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        try:
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except Exception as exc:
            logger.error("Similarity computation error: %s", exc)
            return 0.0

    def analyze(
        self,
        text1: str,
        text2: str,
        chunks1: List[str],
        chunks2: List[str],
        threshold: float = 0.7,
        analysis_mode: str = "Standard",
    ) -> Dict:
        """
        Perform comprehensive similarity analysis.

        Embeddings are computed ONCE for all chunks and reused by both the
        chunk-matching step and the heatmap - no duplicate model inference.
        """
        logger.info("Starting %s analysis (%d x %d chunks)", analysis_mode, len(chunks1), len(chunks2))
        mode_cfg = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["Standard"])

        # Step 1: overall document-level similarity
        emb1_full = self.get_embedding(text1)
        emb2_full = self.get_embedding(text2)
        overall_similarity = self.compute_similarity(emb1_full, emb2_full)

        # Step 2: chunk embeddings (computed ONCE, shared below)
        embeddings1 = self.get_embeddings_batch(chunks1)
        embeddings2 = self.get_embeddings_batch(chunks2)

        # Step 3: pairwise similarity matrix (single computation)
        sim_matrix = cosine_similarity(embeddings1, embeddings2)

        # Step 4: matched chunk pairs via NumPy masking (no Python loop)
        chunk_results = self._match_chunks_numpy(
            chunks1, chunks2, sim_matrix, threshold, top_n=mode_cfg["top_matches"]
        )

        # Step 5: heatmap reuses already-computed sim_matrix
        heatmap_data = self._generate_heatmap(
            sim_matrix, len(chunks1), len(chunks2),
            limit=mode_cfg["heatmap_limit"],
        )

        # Step 6: top matching segment details
        matching_segments = self._format_matching_segments(
            chunks1, chunks2, chunk_results, top_n=mode_cfg["top_matches"]
        )

        # Step 7: chunk-based plagiarism score (the correct formula)
        #
        # For every Doc1 chunk, find its best (maximum) similarity to any Doc2 chunk.
        # Average those per-chunk best scores across all Doc1 chunks.
        #
        # Why this is better than  overall_similarity * 100:
        #   - overall_similarity is a single full-document embedding comparison;
        #     two essays on the same topic score 60-70% even if both are original.
        #   - This formula is grounded in actual content overlap: 0% means no chunk
        #     in Doc1 has any match in Doc2; 100% means every chunk is fully duplicated.
        best_match_per_chunk = sim_matrix.max(axis=1)          # shape: (n_chunks1,)
        plagiarism_score = float(best_match_per_chunk.mean() * 100)

        # unique_percentage: fraction of Doc1 chunks with NO match above threshold
        matched_chunk1_indices = {r["chunk1_idx"] for r in chunk_results}
        unique_percentage = (
            1.0 - len(matched_chunk1_indices) / len(chunks1)
            if chunks1 else 1.0
        )

        # Step 8: supporting metadata
        risk_level = self._determine_risk_level(plagiarism_score / 100)
        doc1_stats = self._calculate_doc_stats(text1)
        doc2_stats = self._calculate_doc_stats(text2)
        interpretation = self._generate_interpretation(
            plagiarism_score / 100, len(matching_segments), risk_level
        )

        results = {
            "overall_similarity": overall_similarity,
            "plagiarism_score": plagiarism_score,          # chunk-based, correct
            "risk_level": risk_level,
            "matching_segments": len(matching_segments),
            "matching_segments_detail": matching_segments,
            # unique_percentage = share of Doc1 chunks with NO similar counterpart in Doc2
            "unique_percentage": unique_percentage,
            "chunk_similarities": chunk_results,
            "heatmap_data": heatmap_data,
            "doc1_stats": doc1_stats,
            "doc2_stats": doc2_stats,
            "interpretation": interpretation,
            "threshold_used": threshold,
            "analysis_mode": analysis_mode,
        }

        logger.info("Analysis complete. Similarity: %.2f%%", overall_similarity * 100)
        return results

    def _match_chunks_numpy(
        self,
        chunks1: List[str],
        chunks2: List[str],
        sim_matrix: np.ndarray,
        threshold: float,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Find above-threshold chunk pairs using NumPy boolean masking.

        Replaces the previous O(n*m) Python for-loop over an already-computed
        matrix with a vectorised operation.
        """
        row_ids, col_ids = np.where(sim_matrix >= threshold)
        scores = sim_matrix[row_ids, col_ids]

        # Sort descending by score, take top_n
        order = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in order:
            i, j = int(row_ids[idx]), int(col_ids[idx])
            results.append({
                "chunk1_idx": i,
                "chunk2_idx": j,
                "chunk1_text": chunks1[i][:200],
                "chunk2_text": chunks2[j][:200],
                "similarity": float(scores[idx]),
            })
        return results

    def _generate_heatmap(
        self,
        sim_matrix: np.ndarray,
        n_chunks1: int,
        n_chunks2: int,
        limit: int = 35,
    ) -> go.Figure:
        """
        Build a Plotly heatmap from the pre-computed similarity matrix.

        Accepts the existing sim_matrix rather than re-encoding chunks,
        eliminating duplicate model inference.
        """
        try:
            r = min(n_chunks1, limit)
            c = min(n_chunks2, limit)
            z = sim_matrix[:r, :c]

            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=[f"C2-{i+1}" for i in range(c)],
                y=[f"C1-{i+1}" for i in range(r)],
                colorscale="RdYlGn",
                colorbar=dict(title="Similarity"),
                hovertemplate="Doc1 Chunk: %{y}<br>Doc2 Chunk: %{x}<br>Similarity: %{z:.2f}<extra></extra>",
            ))
            fig.update_layout(
                title="Chunk-to-Chunk Similarity Heatmap",
                xaxis_title="Document 2 Chunks",
                yaxis_title="Document 1 Chunks",
                height=600,
                width=800,
            )
            return fig
        except Exception as exc:
            logger.error("Heatmap generation error: %s", exc)
            return go.Figure()

    def _format_matching_segments(
        self,
        chunks1: List[str],
        chunks2: List[str],
        chunk_results: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        """Format top matching segments for display."""
        segments = []
        for result in chunk_results[:top_n]:
            i, j = result["chunk1_idx"], result["chunk2_idx"]
            segments.append({
                "text1": chunks1[i][:500],
                "text2": chunks2[j][:500],
                "score": result["similarity"],
                "position1": i,
                "position2": j,
            })
        return segments

    def _determine_risk_level(self, similarity: float) -> str:
        """
        Determine plagiarism risk level
        
        Args:
            similarity: Overall similarity score
            
        Returns:
            Risk level string
        """
        if similarity >= self.thresholds["high"]:
            return "High"
        elif similarity >= self.thresholds["moderate"]:
            return "Moderate"
        else:
            return "Low"

    def _calculate_doc_stats(self, text: str) -> Dict:
        """
        Calculate document statistics
        
        Args:
            text: Document text
            
        Returns:
            Statistics dictionary
        """
        try:
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            return {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": max(sentences, 1),
                "unique_words": len(set(w.lower() for w in words)),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            }
        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return {}

    def _generate_interpretation(
        self, similarity: float, matching_count: int, risk_level: str
    ) -> str:
        """
        Generate human-readable interpretation
        
        Args:
            similarity: Overall similarity score
            matching_count: Number of matching segments
            risk_level: Risk level
            
        Returns:
            Interpretation string
        """
        pct = similarity * 100  # plagiarism_score / 100 is passed in as similarity
        interpretations = {
            "High": (
                f"Plagiarism score of {pct:.1f}% indicates substantial content is shared between the documents. "
                f"Found {matching_count} highly similar chunk pair(s). "
                "Manual review is strongly recommended to verify originality and proper attribution."
            ),
            "Moderate": (
                f"Plagiarism score of {pct:.1f}% with {matching_count} overlapping segment(s). "
                "This level may reflect a shared research area, common phrasing, or paraphrasing. "
                "Review the highlighted matches to determine whether attribution is required."
            ),
            "Low": (
                f"Plagiarism score of {pct:.1f}% with {matching_count} minor overlap(s). "
                "The documents appear largely original. "
                "The detected matches are likely coincidental phrasing or common terminology."
            ),
        }
        
        return interpretations.get(risk_level, "Unable to generate interpretation.")

    def compare_multiple(self, documents: List[str]) -> Dict:
        """
        Compare multiple documents (future feature)
        
        Args:
            documents: List of document texts
            
        Returns:
            Comparison results
        """
        # Placeholder for batch comparison feature
        logger.info("Multi-document comparison not yet implemented")
        return {"error": "Feature coming soon"}