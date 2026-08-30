import pytest

from ego_progress.score import parse_score


def valid_response():
    return '{"reward":80,"task_progress":90,"visual_observability":70,"data_quality":80,"confidence":75,"success_likelihood":85,"failure_mode":null,"evidence":"The object moves onto the table."}'


def test_parse_score_accepts_contract():
    assert parse_score(valid_response())["reward"] == 80


def test_parse_score_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="reward"):
        parse_score(valid_response().replace('"reward":80', '"reward":101'))
