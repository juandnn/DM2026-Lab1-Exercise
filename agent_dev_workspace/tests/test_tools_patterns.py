import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_tool_known_answer_17():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_var["patterns"] == []
    assert "note" in res_var

    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    res_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catA_sup4["patterns"] == []

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_known_answer_20():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 1. Test topk algorithm (FAE) at k=1, k=2, k=3
    for k_val in [1, 2, 3]:
        res_catA_topk = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert len(res_catA_topk["patterns"]) == 1
        assert res_catA_topk["patterns"][0]["pattern"] == ["alpha"]
        assert res_catA_topk["patterns"][0]["support"] == 3

        res_catB_topk = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert len(res_catB_topk["patterns"]) == 1
        assert res_catB_topk["patterns"][0]["pattern"] == ["gamma"]
        assert res_catB_topk["patterns"][0]["support"] == 4

    # 2. Test maxfpgrowth algorithm (MaxFPGrowth) returns 0 patterns for both categories
    for min_s in [1, 4]:
        res_catA_max = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_s,
        })
        assert res_catA_max["patterns"] == []

        res_catB_max = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_s,
        })
        assert res_catB_max["patterns"] == []
