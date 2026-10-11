import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Literal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 2.0,
        n_neighbors: int = 3,
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensions (PCA/t-SNE/UMAP) and plots a 2D scatter plot colored by category.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap'). Defaults to 'pca'.
            n_components: Target dimension count. Defaults to 2.
            random_state: Random seed for reproducible output. Defaults to 42.
            perplexity: Perplexity parameter for t-SNE. Defaults to 2.0.
            n_neighbors: Number of neighbors parameter for UMAP. Defaults to 3.

        Returns:
            Dict[str, Any]: Summary containing result_id, method, n_components, coordinates, and explained_variance_ratio.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is None. Build DTM first."}

        dense_matrix = session.feature_matrix.toarray()
        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_coords = reducer.fit_transform(dense_matrix)
            explained_variance_ratio = [
                float(round(val, 4)) for val in reducer.explained_variance_ratio_.tolist()
            ]
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            reduced_coords = reducer.fit_transform(dense_matrix)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            reduced_coords = reducer.fit_transform(dense_matrix)
        else:
            return {"error": f"Invalid reduction method '{method}'."}

        coords_list = [[float(round(c, 4)) for c in row] for row in reduced_coords.tolist()]

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)
        else:
            categories = ["Unassigned"] * len(dense_matrix)

        plot_df = pd.DataFrame({
            "Dim 1": reduced_coords[:, 0],
            "Dim 2": reduced_coords[:, 1] if n_components > 1 else np.zeros(len(reduced_coords)),
            "Category": categories,
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            data=plot_df,
            ax=ax,
            s=80,
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        session.pending_figure = fig
        session.artifacts[f"reduced_{method}"] = coords_list

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary=summary,
        )

        return summary

    @tool
    def binarize_labels_tool(
        sparse: bool = False,
        drop_first: bool = False,
    ) -> Dict[str, Any]:
        """One-hot encodes category labels in session.dataframe into dummy indicator variables.

        Args:
            sparse: Whether to return sparse dummy DataFrame. Defaults to False.
            drop_first: Whether to drop the first category to get K-1 dummies. Defaults to False.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, n_categories, and categories list.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        cat_col = None
        if "category_name" in session.dataframe.columns:
            cat_col = "category_name"
        elif "category" in session.dataframe.columns:
            cat_col = "category"
        else:
            return {"error": "Neither 'category_name' nor 'category' found in session.dataframe."}

        cat_series = session.dataframe[cat_col]
        encoded_df = pd.get_dummies(cat_series, sparse=sparse, drop_first=drop_first).astype(int)

        categories_encoded = encoded_df.columns.astype(str).tolist()
        session.artifacts["binarized_labels"] = encoded_df

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "n_categories": len(categories_encoded),
            "categories": categories_encoded,
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={"sparse": sparse, "drop_first": drop_first},
            summary=summary,
            full_report=encoded_df,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
