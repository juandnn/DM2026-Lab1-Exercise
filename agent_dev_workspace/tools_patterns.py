import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a single category using PAMI pattern mining algorithms.

        Args:
            category_name: Category to mine documents for.
            filtering_method: Vocabulary filter method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold (for 'fpgrowth' and 'maxfpgrowth'). Defaults to 1.
            k: Number of top patterns to keep (for 'topk'). Defaults to 10.

        Returns:
            Dict[str, Any]: Summary containing result_id, category_name, filtering_method, algorithm, and patterns list.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category_name '{category_name}'."}

        texts = cat_df["text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(stop_words="english")
        try:
            counts_matrix = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        if len(feature_names) == 0:
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        dense_counts = counts_matrix.toarray()

        if filtering_method == "variance":
            scores = np.var(dense_counts, axis=0)
            low_pct, high_pct = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > low_pct) & (scores < high_pct)
        elif filtering_method == "term_frequency":
            scores = np.sum(dense_counts, axis=0)
            low_pct, high_pct = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > low_pct) & (scores < high_pct)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(counts_matrix).toarray()
            scores = np.mean(tfidf_matrix, axis=0)
            pct_20 = np.percentile(scores, 20)
            mask = scores >= pct_20
        else:
            return {"error": f"Invalid filtering_method '{filtering_method}'."}

        surviving_terms = feature_names[mask].tolist()

        if len(surviving_terms) == 0:
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []

            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result(
                tool_name="mine_patterns_tool",
                args={
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary=summary,
            )
            return summary

        surviving_indices = [vectorizer.vocabulary_[term] for term in surviving_terms]
        filtered_dense = dense_counts[:, surviving_indices]

        binary_df = pd.DataFrame(
            (filtered_dense >= 1).astype(int),
            columns=surviving_terms,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = os.path.join(temp_dir, "db.txt")
            converter = DF2DB.DF2DB(binary_df, "dense")
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_file, minSup=min_sup)
            elif algorithm == "topk":
                miner = FAE.FAE(db_file, k=k)
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
            else:
                return {"error": f"Invalid algorithm '{algorithm}'."}

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

        patterns_list = []
        if patterns_df is not None and not patterns_df.empty:
            for _, row in patterns_df.iterrows():
                pattern_str = str(row.iloc[0])
                pattern_items = [item.strip() for item in pattern_str.replace("\t", " ").split() if item.strip()]
                support_val = int(float(row.iloc[1]))
                patterns_list.append({"pattern": pattern_items, "support": support_val})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list,
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary=summary,
        )

        return summary

    return [mine_patterns_tool]
