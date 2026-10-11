import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two document row vectors in session.feature_matrix.

        Args:
            doc_index_1: Row index of the first document (0-based).
            doc_index_2: Row index of the second document (0-based).

        Returns:
            Dict[str, Any]: Summary containing result_id, doc_index_1, doc_index_2, and similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is None. Build DTM first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"doc_index_1 {doc_index_1} is out of bounds for {n_docs} documents."}
        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"doc_index_2 {doc_index_2} is out of bounds for {n_docs} documents."}

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        sim_matrix = cosine_similarity(vec1, vec2)
        raw_sim = float(sim_matrix[0, 0])

        if np.isnan(raw_sim):
            similarity_val = 0.0
        else:
            similarity_val = float(round(raw_sim, 4))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": similarity_val,
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary=summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation heatmap over top terms ordered by variance.

        Args:
            top_k: Number of highest variance terms to include in correlation matrix. Defaults to 20.

        Returns:
            Dict[str, Any]: Summary containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        dense_matrix = session.feature_matrix.toarray().astype(float)
        variances = np.var(dense_matrix, axis=0)

        total_terms = len(session.feature_names)
        k = min(top_k, total_terms)

        top_indices = np.argsort(variances)[::-1][:k]

        top_terms = [session.feature_names[i] for i in top_indices]
        submatrix = dense_matrix[:, top_indices]

        corr_matrix = np.corrcoef(submatrix, rowvar=False)

        if k == 1:
            corr_matrix = np.array([[1.0]])

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="vlag",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        matrix_list = [[float(round(cell, 4)) for cell in row] for row in corr_matrix.tolist()]

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": matrix_list,
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_k": top_k},
            summary=summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
