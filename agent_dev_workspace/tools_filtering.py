import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Any, Optional
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Flags near-constant terms by variance in session.feature_matrix for removal.

        Args:
            threshold: Variance threshold below which terms are flagged for removal. Defaults to 0.1.

        Returns:
            Dict[str, Any]: Summary containing result_id, threshold, total_terms,
            kept_terms_count, removed_terms_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)

        term_var_pairs = list(zip(session.feature_names, variances.tolist()))

        sorted_pairs = sorted(term_var_pairs, key=lambda x: x[1], reverse=True)
        full_df = pd.DataFrame(sorted_pairs, columns=["term", "variance"])

        kept_terms = [t for t, v in term_var_pairs if v >= threshold]
        removed_terms = [t for t, v in term_var_pairs if v < threshold]

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=full_df,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str = "catB",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes Pearson correlation between terms in session.feature_matrix and target class.

        Args:
            target_class: Category name to evaluate correlation against. Defaults to "catB".
            threshold: Optional threshold for filtering terms by absolute Pearson correlation |r|.

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, total_terms, and correlations dict.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe is None or 'category_name' column is missing."}

        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        y_std = np.std(y)
        dense_matrix = session.feature_matrix.toarray().astype(float)

        term_corr_pairs = []
        correlations_dict = {}

        for i, term in enumerate(session.feature_names):
            x_col = dense_matrix[:, i]
            x_std = np.std(x_col)

            if x_std == 0 or y_std == 0:
                r_val = 0.0
            else:
                r_val = float(pearsonr(x_col, y)[0])
                if np.isnan(r_val):
                    r_val = 0.0

            r_val_rounded = round(r_val, 4)
            term_corr_pairs.append((term, r_val_rounded))
            correlations_dict[term] = r_val_rounded

        sorted_pairs = sorted(term_corr_pairs, key=lambda item: item[1], reverse=True)
        full_df = pd.DataFrame(sorted_pairs, columns=["term", "pearson_r"])

        kept_terms = []
        removed_terms = []
        if threshold is not None:
            for t, r in term_corr_pairs:
                if abs(r) >= threshold:
                    kept_terms.append(t)
                else:
                    removed_terms.append(t)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "correlations": correlations_dict,
        }

        if threshold is not None:
            summary["kept_terms"] = kept_terms
            summary["removed_terms"] = removed_terms

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_df,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str = "catB",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes Spearman rank correlation between terms in session.feature_matrix and target class.

        Args:
            target_class: Category name to evaluate correlation against. Defaults to "catB".
            threshold: Optional threshold for filtering terms by absolute Spearman correlation |r_s|.

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, total_terms, and correlations dict.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Build DTM first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe is None or 'category_name' column is missing."}

        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        y_std = np.std(y)
        dense_matrix = session.feature_matrix.toarray().astype(float)

        term_corr_pairs = []
        correlations_dict = {}

        for i, term in enumerate(session.feature_names):
            x_col = dense_matrix[:, i]
            x_std = np.std(x_col)

            if x_std == 0 or y_std == 0:
                r_val = 0.0
            else:
                r_val = float(spearmanr(x_col, y)[0])
                if np.isnan(r_val):
                    r_val = 0.0

            r_val_rounded = round(r_val, 4)
            term_corr_pairs.append((term, r_val_rounded))
            correlations_dict[term] = r_val_rounded

        sorted_pairs = sorted(term_corr_pairs, key=lambda item: item[1], reverse=True)
        full_df = pd.DataFrame(sorted_pairs, columns=["term", "spearman_r"])

        kept_terms = []
        removed_terms = []
        if threshold is not None:
            for t, r in term_corr_pairs:
                if abs(r) >= threshold:
                    kept_terms.append(t)
                else:
                    removed_terms.append(t)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "correlations": correlations_dict,
        }

        if threshold is not None:
            summary["kept_terms"] = kept_terms
            summary["removed_terms"] = removed_terms

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_df,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
