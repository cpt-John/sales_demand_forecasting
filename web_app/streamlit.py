from datetime import date, datetime
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from hybrid_model import predict, unique_items

st.set_page_config(
    page_icon="📊", page_title="Sales Forecast App", layout="wide")


def plot_all(
    source, x="date", y="sales", group="item", axis_scale="linear",
):

    brush = alt.selection_interval(encodings=["x"], empty="all", )

    click = alt.selection_multi(encodings=["color"], )

    lines = (
        (
            alt.Chart(source,)
            .mark_line(point=True)
            .encode(
                x=x,
                y=alt.Y(y, scale=alt.Scale(type=f"{axis_scale}")),
                color=group,
                tooltip=[
                    x,
                    group,
                    y,
                    alt.Tooltip("delta", format=".2%"),
                ],
            )
        )
        .add_selection(brush)
        .properties(width=550)
        .transform_filter(click)
    )

    bars = (
        alt.Chart(source, )
        .mark_bar()
        .encode(
            y=group,
            color=group,
            x=alt.X(f"{y}:Q", scale=alt.Scale(type=f"{axis_scale}")),
            tooltip=[x, y, alt.Tooltip("delta", format=".2%")],
        )
        .transform_filter(brush)
        .properties(width=550)
        .add_selection(click)
    )

    return lines & bars


start_date, no_of_days, predict_items, rolling_avg_days = \
    ["2020-1-1", 366, ["1", "2", "5"], 1]


def process_df():
    df = predict(start_date, no_of_days, list(map(int, predict_items)))
    df["sales"] = df["total"]
    df["date"] = df.index
    df = df.astype({"item": "str"})
    df["delta"] = (df.groupby(["item"])["sales"].pct_change()).fillna(0)

    final_df = df.groupby(["item"]).rolling(rolling_avg_days).mean(numeric_only=True).dropna()
    final_df = final_df.reset_index()
    final_df.set_index("date", inplace=True,  drop=False,)
    return final_df


st.markdown(
    '<style>.block-container{padding-top: 1.3em;padding-bottom: 2em;}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-top:0; padding:0'>Item Sales Prediction</h1>",
            unsafe_allow_html=True)
st.markdown(" <p style='text-align: center'> This app predicts the data using a hybrid model \
| The csv file can be found along with the source code \
| There is also a EDA notebook along with the source</p>", unsafe_allow_html=True)
st.markdown(
    """<div style='display: flex;flex-direction: row;justify-content:center'><a href="https://github.com/cpt-John/sales_demand_forecasting">\
        Source Code</a><div style='padding-inline:0.5em'> | </div><a href="https://cpt-john.github.io/">Portfolio</a></div>
    """, unsafe_allow_html=True
)
r1c1, r1c2, r1c3 = st.columns((1, 3, 3))
r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns((1, 1.5, 1.5, 1.5, 1.5))

with r1c1:
    st.markdown("**Prediction settings**")
    start_date = st.date_input(
        "Select start date",
        date(2020, 1, 1),
        min_value=datetime.strptime("2017-01-01", "%Y-%m-%d"),
        max_value=datetime.now(),
    )
    predict_items = st.multiselect(
        "Select Items to predict",
        unique_items,
        default=["1"],
    )
    no_of_days = st.slider("Days to predict", min_value=1,
                           max_value=1000, value=365, step=1,)

    rolling_avg_days = st.slider("Mean over days", min_value=1,
                                 max_value=60, value=1, step=1,)

with r1c1:
    st.markdown(" <ul > \
    <li style='font-size: 0.6em'>Click on a bar to highlight that item</li>\
    <li style='font-size: 0.6em'>Click and drag line chart to select and pan date interval</li>\
    <li style='font-size: 0.6em'>Hover over bar chart to view sales</li>\
    </ul>", unsafe_allow_html=True)

with r1c2:
    st.header("Compare Items")
df = process_df()
item_names = df["item"].unique()


with r2c2:
    select_items = st.multiselect(
        "Select Items to compare",
        item_names,
        default=item_names[:3],
    )
    if len(select_items) <= 3:
        pass
    else:
        st.warning("You can select max 3 items")
        st.stop()
    if not select_items:
        st.stop()

select_items_df = pd.DataFrame(
    select_items).rename(columns={0: "item"})
filtered_df = df[
    df["item"].isin(select_items_df["item"])
]

with r1c2:
    # Assuming your plot_all function returns an Altair chart
    chart = plot_all(filtered_df, x="date", y="sales")

    # Specify encoding with channel types
    chart = chart.encode(
        x=alt.X("date:T", title="Date"),  # "T" specifies a temporal field
        y=alt.Y("sales:Q", title="Sales"),  # "Q" specifies a quantitative field
    )
    st.altair_chart(chart, use_container_width=True)

with r1c3:
    st.header("Hybrid Model Components")

with r2c4:
    selected_component_item = st.selectbox(
        "Select Item to decompose",
        select_items,
        index=0,)

hybrid_models_info = """
rf   - Random Forest (residual)\n
poly - Polynomial (weekly trend)\n
fft  - Fourier Transform (yearly trend)\n
reg  - Linear Regression (overall trend)
"""
with r2c5:
    summation = st.checkbox("View Summation")
    select_components = st.multiselect(
        "Select Components to add",
        ['rf', "poly", "fft", "reg", ],
        default=[],
        help=hybrid_models_info
    )
componets_df = filtered_df[filtered_df['item'] == selected_component_item]
value_vars = ['rf', "poly", "fft", "reg", ]
if len(select_components) and summation:
    componets_df["total"] = componets_df[select_components].sum(axis=1)
    value_vars = value_vars+["total"]
componets_df = pd.melt(componets_df, id_vars=['date'], value_vars=value_vars,
                       var_name='model', value_name='sales').astype({"model": "str"}).set_index("date")
componets_df["delta"] = componets_df.groupby(
    ["model"])["sales"].pct_change().fillna(0)
componets_df['date'] = componets_df.index

# with r1c3:
#     st.altair_chart(plot_all(componets_df,x = "date", y="sales", group="model"),
#                     use_container_width=True)
