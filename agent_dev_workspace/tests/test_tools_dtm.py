import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    res_default = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    expected_terms = ["alpha", "always", "beta", "delta", "gamma"]

    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5
    assert res_default["terms"] == expected_terms
    assert res_default["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    expected_sub_matrix = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    assert res_slice["n_documents"] == 3
    assert res_slice["n_terms"] == 2
    assert res_slice["terms"] == ["alpha", "always"]
    assert res_slice["matrix_slice"] == expected_sub_matrix
