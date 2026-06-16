# visualization/charts.py
import plotly.express as px
import plotly.graph_objects as go

def line_chart(df, x, y, title=""):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    return fig

def bar_chart(df, x, y, title=""):
    fig = px.bar(df, x=x, y=y, title=title)
    return fig

def scatter_chart(df, x, y, color=None, title=""):
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    return fig

def multi_line(df, x, ys, title=""):
    fig = go.Figure()
    for y in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=y))
    fig.update_layout(title=title)
    return fig
