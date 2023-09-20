from heartbeatpy import (
    calculate_votes,
    create_sample_data,
    calculate_summary,
    visualize_heartbeat,
)
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    sample_data = create_sample_data()
    return sample_data


@pytest.fixture
def sample_summary():
    sample_data = create_sample_data()
    votes, _ = calculate_votes(data=sample_data[0], columns=sample_data[1])
    summary = calculate_summary(votes, columns=sample_data[1])
    return summary


def test_vote_calc():
    data = pd.DataFrame(
        {
            "q1": [5, 4, 5, 4, 5, 4],
            "q2": [4, 4, 5, 4, 4, 4],
            "q3": [5, 4, 5, 4, 4, 5],
            "q4": [5, 3, 5, 4, 5, 5],
            "q5": [4, 4, 5, 4, 5, 4],
        }
    )

    expected_output_votes = pd.DataFrame(
        {
            "q1": [0, 0, 0, 0, 0, 0],
            "q2": [-1, 0, 0, 0, -1, 0],
            "q3": [0, 0, 0, 0, -1, 1],
            "q4": [0, -1, 0, 0, 0, 1],
            "q5": [-1, 0, 0, 0, 0, 0],
        }
    )

    expected_output_stats = pd.DataFrame(
        {
            "individual_sd": [0.5477226, 0.4472136, 0, 0, 0.5477226, 0.5477226],
            "individual_mean": [4.6, 3.8, 5, 4, 4.6, 4.4],
        }
    )

    votes, stats = calculate_votes(data, columns=data.columns)
    pd.testing.assert_frame_equal(left=votes, right=expected_output_votes)
    pd.testing.assert_frame_equal(left=stats, right=expected_output_stats)


def test_provide_threshold(sample_data):
    calculate_votes(data=sample_data[0], columns=sample_data[1], threshold=0.5)


def test_zero_stdev():
    data = pd.DataFrame(
        {
            "q1": [5] * 5,
            "q2": [5] * 5,
            "q3": [5] * 5,
            "q4": [5] * 5,
            "q5": [5] * 5,
        }
    )
    _, stats = calculate_votes(data, columns=data.columns)
    assert (stats["individual_sd"] == 0).all()


def test_nonnumeric_cols(sample_data):
    df = sample_data[0]
    df["text_col"] = "Strongly Agree"

    with pytest.raises(TypeError):
        calculate_votes(data=df, columns=["text_col"])


def test_negative_threshold(sample_data):
    with pytest.raises(ValueError):
        calculate_votes(data=sample_data[0], columns=sample_data[1], threshold=-1)


def test_viz(sample_summary):
    visualize_heartbeat(sample_summary)


def test_incorrect_viz_sort_col(sample_summary):
    with pytest.raises(IndexError):
        # 'test' column to sort by not present in summary df
        visualize_heartbeat(sample_summary, sort_by="test")


def test_incorrect_viz_question_col(sample_summary):
    with pytest.raises(IndexError):
        # 'test' column to use as question col not present in summary df
        visualize_heartbeat(sample_summary, question_col="test")


def test_viz_missing_cols(sample_summary):
    summary_subset = sample_summary[["Question", "Upvote", "Downvote"]]

    with pytest.raises(ValueError):
        # summary_subset df does not have all required columns (missing "Neutral")
        visualize_heartbeat(summary_subset)
