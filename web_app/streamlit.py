from datetime import date, datetime
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from hyper_model import predict, unique_items

st.set_page_config(page_icon="📥", page_title="Download App")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


def main():

    # Note that page title/favicon are set in the __main__ clause below,
    # so they can also be set through the mega multipage app (see ../pandas_app.py).

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Select start date",
            date(2020, 1, 1),
            min_value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
            max_value=datetime.now(),
        )

    with col2:
        time_frame = st.selectbox(
            "Select weekly or monthly downloads", ("weekly", "monthly")
        )


def plot_all_downloads(
    source, x="date", y="sales", group="item", axis_scale="linear"
):

    if st.checkbox("View logarithmic scale"):
        axis_scale = "log"

    brush = alt.selection_interval(encodings=["x"], empty="all")

    click = alt.selection_multi(encodings=["color"])

    lines = (
        (
            alt.Chart(source)
            .mark_line(point=True)
            .encode(
                x=x,
                y=alt.Y("sales", scale=alt.Scale(type=f"{axis_scale}")),
                color=group,
                tooltip=[
                    "date",
                    "item",
                    "sales",
                    alt.Tooltip("delta", format=".2%"),
                ],
            )
        )
        .add_selection(brush)
        .properties(width=550)
        .transform_filter(click)
    )

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            y=group,
            color=group,
            x=alt.X("sales:Q", scale=alt.Scale(type=f"{axis_scale}")),
            tooltip=["date", "sales", alt.Tooltip("delta", format=".2%")],
        )
        .transform_filter(brush)
        .properties(width=550)
        .add_selection(click)
    )

    return lines & bars


col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Select start date",
        date(2020, 1, 1),
        min_value=datetime.strptime("2017-01-01", "%Y-%m-%d"),
        max_value=datetime.now(),
    )
with col2:
    predict_items = st.multiselect(
        "Select Items to compare",
        unique_items,
        default=["1"],
    )

no_of_days = st.slider("Days to predict", min_value=1,
                       max_value=1000, value=365, step=1,)


df = predict(start_date, no_of_days, list(map(int, predict_items)))
df["sales"] = df["total"]
df["date"] = df.index
df = df.astype({"item": "str"})
df["delta"] = (df.groupby(["item"])["sales"].pct_change()).fillna(0)

item_names = df["item"].unique()
st.header("Compare Items")
instructions = """
Click and drag line chart to select and pan date interval\n
Hover over bar chart to view sales\n
Click on a bar to highlight that item
"""
select_items = st.multiselect(
    "Select Items to compare",
    item_names,
    default=item_names[:3],
    help=instructions,
)
if len(select_items) <= 3:
    pass
else:
    st.warning("You can select max 3 items")
    st.stop()

select_items_df = pd.DataFrame(
    select_items).rename(columns={0: "item"})
if not select_items:
    st.stop()
filtered_df = df[
    df["item"].isin(select_items_df["item"])
]

rolling_avg_days = st.slider("Mean over days", min_value=1,
                             max_value=60, value=1, step=1,)
final_df = filtered_df.groupby("item").rolling(
    rolling_avg_days).mean().dropna()
final_df = final_df.reset_index()
final_df.set_index("date", inplace=True,  drop=False,)

st.altair_chart(plot_all_downloads(final_df), use_container_width=True)
