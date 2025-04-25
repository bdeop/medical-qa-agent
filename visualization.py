import plotly.express as px
import pandas as pd

def plot_medical_timeline(conditions_df):
    df = conditions_df.copy()
    df["START"] = pd.to_datetime(df["START"], errors='coerce')
    df["STOP"] = pd.to_datetime(df["STOP"], errors='coerce')
    df = df.sort_values("START", ascending=False).head(20)

    fig = px.timeline(
        df,
        x_start="START", x_end="STOP", y="DESCRIPTION",
        color="CATEGORY" if "CATEGORY" in df else None,
        hover_data=["CODE", "START", "STOP"],
        title="📅 Patient Medical Timeline"
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500)
    return fig

def format_labs_vitals(obs):
    obs = obs[obs["DESCRIPTION"].notnull()]
    df = obs[["DESCRIPTION", "VALUE", "UNITS", "DATE"]].copy()
    df.columns = ["Test/Measure", "Value", "Units", "Date"]
    return df.sort_values("Date", ascending=False)
