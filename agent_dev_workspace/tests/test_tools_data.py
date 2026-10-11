import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert res["n_documents"] == 8
    assert res["total_tokens"] == sum(len(tokens) for tokens in expected_unigrams)
    assert session.dataframe["unigrams"].tolist() == expected_unigrams
    assert session.artifacts["unigrams"] == expected_unigrams


def test_list_files_tool_known_answer_11():
    session = SessionState()

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool_known_answer_2():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    res_check = check_duplicates_tool1.invoke({"drop": False})
    assert res_check["duplicate_count"] == 1
    assert res_check["dropped"] is False
    assert res_check["n_documents_remaining"] == 8

    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    res_drop = check_duplicates_tool2.invoke({"drop": True})
    assert res_drop["duplicate_count"] == 1
    assert res_drop["dropped"] is True
    assert res_drop["n_documents_remaining"] == 6
    assert len(session2.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["n_samples"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(res["sample_records"]) == 4
    assert "sample_data" in session.artifacts


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert overall["min"] == 0
    assert pytest.approx(overall["25%"], abs=1e-3) == 17.75
    assert pytest.approx(overall["50%"], abs=1e-3) == 20.5
    assert pytest.approx(overall["75%"], abs=1e-3) == 25.25
    assert overall["max"] == 30

    by_cat = res["by_category_stats"]
    assert "catA" in by_cat
    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert pytest.approx(catA["mean"], abs=1e-3) == 17.25
    assert pytest.approx(catA["std"], abs=1e-3) == 12.5
    assert catA["min"] == 0
    assert pytest.approx(catA["25%"], abs=1e-3) == 12.75
    assert pytest.approx(catA["50%"], abs=1e-3) == 20.0
    assert pytest.approx(catA["75%"], abs=1e-3) == 24.5
    assert catA["max"] == 29

    assert "catB" in by_cat
    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert pytest.approx(catB["mean"], abs=1e-3) == 22.5
    assert pytest.approx(catB["std"], abs=1e-3) == 5.7446
    assert catB["min"] == 18
    assert pytest.approx(catB["25%"], abs=1e-3) == 18.0
    assert pytest.approx(catB["50%"], abs=1e-3) == 21.0
    assert pytest.approx(catB["75%"], abs=1e-3) == 25.5
    assert catB["max"] == 30

    assert session.pending_figure is not None


def test_inspect_data_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 3, "start_col": 0, "end_col": 2})

    assert res["total_rows"] == 8
    assert len(res["inspected_columns"]) == 2
    assert res["inspected_columns"] == ["text", "category_name"]

    assert len(res["data"]) == 3
    assert res["data"][0] == {"text": "always alpha alpha alpha beta", "category_name": "catA"}
    assert res["data"][1] == {"text": "always alpha alpha beta", "category_name": "catA"}
    assert res["data"][2] == {"text": "always alpha beta", "category_name": "catA"}
