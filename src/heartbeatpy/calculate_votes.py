from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt


def calculate_score(row, col, cols, threshold):
    """
    Calculate a score based on a z-score for a specific value in a DataFrame row.

    Args:
    row (pd.Series): A pandas Series representing a row in a DataFrame.
    col (str): The name of the column for which the z-score should be calculated.
    cols (list of str): A list of column names used to calculate the mean and standard deviation.
    threshold (float): The threshold value for classifying the z-score.

    Returns:
    int: A score based on the z-score and threshold. Returns 1 if z-score is above threshold,
         -1 if z-score is below -threshold, 0 otherwise. If row_std is 0, returns 0. If row[col] is NaN, returns NaN.
    """

    row_mean = row[cols].mean(skipna=True)
    row_std = row[cols].std(skipna=True)
    if row_std == 0:
        return 0
    else:
        zscore = (row[col] - row_mean) / row_std

    if np.isnan(row[col]):
        return np.NaN
    elif threshold < zscore:
        return 1
    elif (-1 * threshold) > zscore:
        return -1
    else:
        return 0


def create_sample_data(ncols: int = 10, nrows: int = 10) -> Tuple[pd.DataFrame, List]:
    """
    Creates a sample survey DataFrame with random integer values.

    Args:
        ncols (int): The number of columns in the DataFrame. Default is 10.
        nrows (int): The number of rows in the DataFrame. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame with an 'ID' column, 'ncols' number of columns
        with random values 1-5 and 'nrows' number of rows.
        List: A list of names of the Likert columns created

    Example:
        >>> df, cols = create_sample_data(ncols=5, nrows=5)
        df.head()
           ID  Q1  Q2  Q3  Q4  Q5
        0   1   4   2   4   2   3
        1   2   4   5   5   1   3
        2   3   2   5   2   5   2
        3   4   1   1   3   1   3
        4   5   5   3   2   3   5
        print(cols)
        ["Q1", "Q2", "Q3", "Q4", "Q5"]
    """
    data = {"ID": range(1, nrows + 1)}

    random_data = np.random.randint(1, 6, size=(nrows, ncols))
    column_names = [f"Q{i}" for i in range(1, ncols + 1)]

    sample_data = pd.DataFrame(
        data=np.column_stack((data["ID"], random_data)), columns=["ID"] + column_names
    )

    return sample_data, column_names


def calculate_votes(
    data: pd.DataFrame, columns: List[str], threshold: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates upvotes and downvotes based on the specified threshold.

    Args:
        data (DataFrame): The input survey data containing the Likert columns of interest.
        columns (List[str]): A list of column names in the dataset to be processed.
        threshold (float, optional): The threshold value used for creating votes. Default is 1.0.

    Returns:
        DataFrame: A modified copy of the input dataset with the specified columns'
        values replaced with votes.
        DataFrame: A dataframe with columns 'individual_mean' containing the mean
        of the columns specified, and 'individual_sd' containing the standard deviation
        of the columns specified.

    Example:
        import pandas as pd
        import numpy as np

        # Create a sample dataset
        data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5]
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 1],
            'C': [3, 4, 5, 6, 1],
            'D': [4, 5, 6, 2, 4],
            'E': [5, 6, 2, 1, 4]
        })

        # Define the columns to be processed
        columns = ['A', 'B', 'C', 'D', 'E']

        # Call the calculate_votes function
        votes, stats = calculate_votes(data, columns, threshold=1)

    """  # noqa: E501
    if threshold <= 0:
        raise ValueError(
            "Threshold must be a positive number of standard deviations"
            " above the mean to define an upvote."
        )

    if not isinstance(columns, list):
        # in case a pandas Index gets passed
        columns = list(columns)
    likert_cols = data[columns]

    # Check if all the values passed are integers, or NAN values since likert
    # data must be an integer range (e.g. 1-5)
    def check_integer_or_nan(column):
        return column.apply(lambda x: (x % 1 == 0) or pd.isna(x)).all()

    all_int_bool = all([check_integer_or_nan(data[col]) for col in columns])

    if not all_int_bool:
        raise TypeError("Non-integer columns passed")

    # Create mean and standard deviation metadata
    stats = pd.DataFrame()
    stats["individual_sd"] = likert_cols.std(axis=1, skipna=True)
    stats["individual_mean"] = likert_cols.mean(axis=1, skipna=True)

    # Calculate votes data
    votes = data.copy()
    for col in columns:
        votes[col] = data.apply(
            lambda row: calculate_score(row, col, columns, threshold), axis=1
        )

    return votes, stats


def percent_neutral(series):
    series = series.dropna()
    return (series == 0).sum() / len(series)


def percent_upvote(series):
    series = series.dropna()
    return (series == 1).sum() / len(series)


def percent_downvote(series):
    series = series.dropna()
    return (series == -1).sum() / len(series)


def calculate_summary(votes: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate the summary of heartbeat analysis for each question in the given dataset.

    Args:
        votes (DataFrame): The dataset containing the votes, created by calculate_votes().
        columns (list): A list of likert question column names in the votes dataset.

    Returns:
        DataFrame: A summary DataFrame with the following columns:
            - Question: The question number.
            - Upvote: The percentage of upvotes for the question.
            - Downvote: The percentage of downvotes for the question.
            - Neutral: The percentage of neutral votes for the question.
            - Controversy_Score: The calculated controversy score for the question.
    """  # noqa: E501

    # Check dataframe only consists of upvotes/downvotes/zeros:
    vote_values = [-1, 0, 1]

    if not all([all(votes[col].isin(vote_values)) for col in columns]):
        raise ValueError(f"Vote dataframe contains values other than {vote_values}")

    summary = (
        votes.melt(value_vars=columns, var_name="Question", value_name="vote")
        .groupby(["Question"])
        .agg(
            **{
                "Upvote": ("vote", percent_upvote),
                "Downvote": ("vote", percent_downvote),
                "Neutral": ("vote", percent_neutral),
            }
        )
        .reset_index()
    )

    summary["Controversy_Score"] = (
        (summary["Upvote"] + summary["Downvote"])
        - abs((summary["Upvote"] - summary["Downvote"]))
    ) / summary["Neutral"]

    return summary


def visualize_heartbeat(
    summary: pd.DataFrame,
    title: str = "Heartbeat Analysis",
    question_col: str = "Question",
    y_axis_title: str = "Question",
    sort_by: str = None,
    sort_ascending: bool = True,
) -> alt.vegalite.v5.api.Chart:
    """Visualize Heartbeat

    Generates a stacked bar chart to visualize the results of heartbeat analysis.
    The function takes in a summary dataframe generated by calculate_summary()
    containing columns [question_col, "Downvote", "Neutral", "Upvote", "Percentage"],
    and optional parameters for customization.

    Args:

        summary (pd.DataFrame): The summary dataframe containing the heartbeat
            analysis data. This can be generated by the function calculate_summary().
        title (str): The title of the chart (default: "Heartbeat Analysis").
        question_col (str): The name of the column in the summary dataframe that
            contains the question names to be displayed on the y-axis
            (default: "Question").
        y_axis_title (str): The title of the y-axis (default: "Question").
        sort_by Optional[str]: The column in the input dataframe to sort the questions
            by (default: None).
        sort_ascending Optional[bool]: Specifies the sort order of the questions
            (default: True). Not used if no `sort_by` column is provided.

    Returns:
        alt.vegalite.v5.api.Chart: The generated bar chart as an Altair chart object.
    """

    if question_col not in summary.columns:
        raise IndexError(f"{question_col} not found in summary DataFrame")
    if sort_by and sort_by not in summary.columns:
        raise IndexError("sort_by column not present in summary DataFrame")

    expected_columns = ["Downvote", "Neutral", "Upvote"]
    found_columns = [col for col in summary if col in expected_columns]

    if not set(expected_columns) == set(found_columns):
        raise ValueError(
            f"Summary table must include the following columns: {expected_columns}",
        )

    # Generate sort order list for visual
    if sort_by:
        questions_sort_order = summary.sort_values(sort_by, ascending=sort_ascending)[
            question_col
        ].tolist()
    else:
        questions_sort_order = summary[question_col].tolist()

    summary_melted = summary.melt(
        id_vars=[question_col],
        value_vars=expected_columns,
        var_name="Type",
        value_name="Percentage",
    )

    color_scale = alt.Scale(
        domain=expected_columns,
        range=["#c30d24", "#cccccc", "#1770ab"],
    )

    y_axis = alt.Axis(
        title=y_axis_title,
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False,
    )

    chart = (
        alt.Chart(summary_melted, title=title)
        .mark_bar()
        .encode(
            alt.X(
                "Percentage", axis=alt.Axis(format="%"), scale=alt.Scale(domain=(0, 1))
            ),
            y=alt.Y(f"{question_col}:N", sort=questions_sort_order).axis(y_axis),
            color=alt.Color("Type:N").title("Response").scale(color_scale),
            tooltip=[
                f"{question_col}",
                "Type",
                alt.Tooltip("Percentage", format=".1%"),
            ],
        )
    )

    return chart
