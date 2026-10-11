import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        min_df: float = 1,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from session.dataframe text.

        Populates session.feature_matrix, session.feature_names, and
        session.artifacts['count_vectorizer'], and reports matrix sparsity.

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for
                different n-grams to be extracted. List of 2 integers, e.g., [1, 1].
            max_features: If not None, build a vocabulary that only considers the
                top max_features ordered by term frequency across the corpus.
            min_df: Ignore terms that have a document frequency strictly lower
                than the given threshold.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents,
            n_terms, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        text_data = session.dataframe["text"].fillna("").astype(str)

        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_features=max_features,
            min_df=min_df,
        )

        feature_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
            },
            summary=summary,
        )

        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Optional limit on the number of top frequent terms to return in the summary.

        Returns:
            Dict[str, Any]: Summary containing result_id, total_terms, and top_terms list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_freqs = list(zip(session.feature_names, counts.tolist()))
        term_freqs.sort(key=lambda x: x[1], reverse=True)

        full_df = pd.DataFrame(term_freqs, columns=["term", "frequency"])

        if top_n is not None and top_n > 0:
            summary_terms = [{"term": t, "frequency": int(f)} for t, f in term_freqs[:top_n]]
        else:
            summary_terms = [{"term": t, "frequency": int(f)} for t, f in term_freqs]

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "top_terms": summary_terms,
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_df,
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Generates a heatmap visualization of a slice of the global document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include in the slice. Defaults to 20.
            n_documents: Number of documents (rows) to include in the slice. Defaults to 20.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, n_terms, matrix_slice, terms, and doc_ids.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        max_docs, max_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        sliced_matrix = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        sliced_terms = session.feature_names[:actual_terms]
        doc_ids = [f"Doc {i}" for i in range(actual_docs)]

        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.5), max(4, actual_docs * 0.4)))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=doc_ids,
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Heatmap (Slice)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        matrix_list = sliced_matrix.tolist()

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "matrix_slice": matrix_list,
            "terms": sliced_terms,
            "doc_ids": doc_ids,
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
