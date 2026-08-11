import pytest

from printqc.parsing import ParseError, parse_model_output


def test_parse_json_label_and_severity():
    result = parse_model_output('{"label":"under_extrusion","severity":2,"confidence":0.7,"evidence":"gaps"}')

    assert result["label"] == "under_extrusion"
    assert result["severity"] == 2


def test_parse_failure_abstains():
    with pytest.raises(ParseError):
        parse_model_output("not json")


def test_conflicting_label_is_rejected():
    with pytest.raises(ParseError, match="label"):
        parse_model_output('{"label":"broken","severity":1}')


def test_parse_trained_chinese_text_under_extrusion():
    result = parse_model_output("类别:欠挤出;严重度:中等(2)。依据:线间空隙明显。")

    assert result["label"] == "under_extrusion"
    assert result["severity"] == 2
    assert result["evidence"] == "线间空隙明显。"


def test_parse_trained_chinese_text_normal():
    result = parse_model_output("类别:正常;严重度:无(0)。依据:表面连续。")

    assert result["label"] == "normal"
    assert result["severity"] == 0
