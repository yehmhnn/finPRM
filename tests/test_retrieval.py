from finprm.retrieval.index import format_demonstrations, retrieval_text


def record(split="train", finqa_id="q1", label=1):
    return {
        "input": {
            "question": "What was the change?",
            "supporting_facts": ["The values were 10 and 12."],
            "prefix": ["subtract(12, 10)"],
            "candidate": "divide(#0, 10)",
        },
        "target": {"label": label},
        "metadata": {"split": split, "finqa_id": finqa_id, "stable_id": finqa_id + "-x"},
    }


def test_retrieval_text_variants():
    item = record()
    assert retrieval_text(item, "question") == "What was the change?"
    joint = retrieval_text(item, "joint")
    assert "subtract(12, 10)" in joint
    assert "divide(#0, 10)" in joint


def test_demonstrations_are_labeled_but_have_no_reward_marker():
    text = format_demonstrations([record(label=0), record(finqa_id="q2", label=1)])
    assert "INCORRECT" in text and "CORRECT" in text
    assert "<extra_0>" not in text
