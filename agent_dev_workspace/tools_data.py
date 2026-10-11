import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into unigrams using nltk.word_tokenize.

        Populates session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, and total_tokens.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is None or 'text' column is missing."}

        def tokenize_text(text: Any) -> list:
            if pd.isna(text) or text is None:
                return []
            s_text = str(text).strip()
            if not s_text:
                return []
            return nltk.word_tokenize(s_text.lower())

        unigrams = session.dataframe["text"].apply(tokenize_text).tolist()

        session.dataframe["unigrams"] = unigrams
        session.artifacts["unigrams"] = unigrams

        n_docs = len(unigrams)
        total_tokens = sum(len(doc_tokens) for doc_tokens in unigrams)

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "total_tokens": total_tokens,
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ".",
    ) -> Dict[str, Any]:
        """Lists files and directories in the specified workspace subdirectory.

        Args:
            subdirectory: Relative path inside the repo to list. Defaults to ".".

        Returns:
            Dict[str, Any]: Summary containing subdirectory, directories list, and files list.
        """
        target_path = os.path.abspath(subdirectory)

        if not os.path.exists(target_path):
            return {"error": f"Path '{subdirectory}' does not exist."}

        if not os.path.isdir(target_path):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories = []
        files = []

        for entry in os.scandir(target_path):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
        )

        return summary

    @tool
    def check_missing_tool(
        column: str = "text",
    ) -> Dict[str, Any]:
        """Checks session.dataframe for missing or empty string values in a column.

        Args:
            column: Name of the column to check for missing/empty values. Defaults to "text".

        Returns:
            Dict[str, Any]: Summary containing result_id, total_rows, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        if column not in session.dataframe.columns:
            return {"error": f"Column '{column}' not found in session.dataframe."}

        col_series = session.dataframe[column]

        missing_indices = []
        for idx, val in col_series.items():
            if pd.isna(val) or val is None or str(val).strip() == "":
                missing_indices.append(int(idx))

        total_rows = len(session.dataframe)
        missing_count = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"column": column},
            summary=summary,
        )

        return summary

    @tool
    def check_duplicates_tool(
        subset: str = "text",
        drop: bool = False,
    ) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops all copies.

        Args:
            subset: Column name to evaluate duplicates on. Defaults to "text".
            drop: If True, drops all copies of duplicated rows in place using keep=False. Defaults to False.

        Returns:
            Dict[str, Any]: Summary containing result_id, duplicate_count, dropped, and n_documents_remaining.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        if subset not in session.dataframe.columns:
            return {"error": f"Column '{subset}' not found in session.dataframe."}

        duplicate_count = int(session.dataframe.duplicated(subset=[subset], keep="first").sum())

        if drop and duplicate_count > 0:
            session.dataframe.drop_duplicates(subset=[subset], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        n_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "dropped": drop,
            "n_documents_remaining": n_remaining,
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"subset": subset, "drop": drop},
            summary=summary,
        )

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Subsamples session.dataframe and returns representative sampled records.

        Args:
            n: Number of rows to sample. Defaults to 5.
            random_state: Random seed for reproducible sampling. Defaults to 42.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_samples, sampled_indices, and sample_records.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        actual_n = min(n, len(session.dataframe))
        sample_df = session.dataframe.sample(n=actual_n, random_state=random_state)

        sampled_indices = [int(i) for i in sample_df.index.tolist()]
        session.artifacts["sample_data"] = sample_df

        records = sample_df.to_dict(orient="records")

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_samples": actual_n,
            "sampled_indices": sampled_indices,
            "sample_records": records,
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
            full_report=sample_df,
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name",
    ) -> Dict[str, Any]:
        """Computes summary statistics on document text length and creates a boxplot by category.

        Args:
            text_column: Column containing document text. Defaults to "text".
            category_column: Column containing document category names. Defaults to "category_name".

        Returns:
            Dict[str, Any]: Summary containing result_id, overall_stats, and by_category_stats.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        temp_df = session.dataframe.copy()
        temp_df["text_length"] = temp_df[text_column].fillna("").astype(str).apply(len)

        overall_desc = temp_df["text_length"].describe().to_dict()
        overall_stats = {k: float(v) for k, v in overall_desc.items()}

        by_category_stats = {}
        if category_column in temp_df.columns:
            cat_groups = temp_df.groupby(category_column)["text_length"]
            for cat_name, group in cat_groups:
                cat_desc = group.describe().to_dict()
                by_category_stats[str(cat_name)] = {k: float(v) for k, v in cat_desc.items()}

        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in temp_df.columns:
            sns.boxplot(x=category_column, y="text_length", data=temp_df, ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y="text_length", data=temp_df, ax=ax)

        ax.set_title("Document Text Length Boxplot")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={"text_column": text_column, "category_column": category_column},
            summary=summary,
        )

        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        start_col: int = 0,
        end_col: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Peeks at the top n rows and a slice of columns in session.dataframe.

        Args:
            n_rows: Number of top rows to inspect. Defaults to 5.
            start_col: Starting column index (0-based). Defaults to 0.
            end_col: Ending column index (0-based, exclusive). If None, inspects through last column.

        Returns:
            Dict[str, Any]: Summary containing result_id, total_rows, total_columns, inspected_columns, and data.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        total_rows, total_cols = session.dataframe.shape

        if start_col < 0 or start_col >= total_cols:
            return {"error": f"start_col {start_col} is out of bounds for {total_cols} columns."}

        actual_end_col = total_cols if end_col is None else min(end_col, total_cols)
        if start_col >= actual_end_col:
            return {"error": f"start_col {start_col} must be less than end_col {actual_end_col}."}

        target_cols = session.dataframe.columns[start_col:actual_end_col].tolist()
        sliced_df = session.dataframe[target_cols].head(n_rows)
        records = sliced_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "inspected_columns": target_cols,
            "data": records,
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows, "start_col": start_col, "end_col": end_col},
            summary=summary,
            full_report=sliced_df,
        )

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
