import os
import pandas as pd


def load_data(fname, survey_1_dir, survey_2_dir, survey_1_name, survey_2_name):
    """Open and join survey data."""

    df_1 = pd.read_csv(os.path.join(survey_1_dir, fname), sep="\t")
    df_1["source"] = survey_1_name
    df_2 = pd.read_csv(os.path.join(survey_2_dir, fname), sep="\t")
    df_2["source"] = survey_2_name
    df = df_1.append(df_2)

    return df


def plot_display(df, index, columns, aggfunc, values, kind="barh"):
    """Generate horizontal bar chart of a table."""
    _df = df.pivot_table(index=index, columns=columns, aggfunc=aggfunc, values=values)
    _df_sum = _df.sum()
    _df = _df / _df_sum
    _df.plot(kind=kind, alpha=0.6)

    return _df
