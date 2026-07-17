# streamlit_pages/15_Global_Heatmap.py
"""Global Heatmap — Professional Edition."""
import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    st.title("🌡️ Global Heatmap")
    st.markdown("Interactive world map of EV metrics with time animation and continent filtering.")

    if df.empty:
        st.warning("No data.")
        return

    # Metric selection
    metric_options = {
        "ETRI": "ETRI",
        "Penetration (%)": "penetration",
        "EV Sales": "ev_sales",
        "EV Stock": "ev_stock",
        "Growth Rate": "growth_rate",
        "Infrastructure": "charging_points_per_100k",
        "EVQTI": "evqti" if "evqti" in df.columns else None,
    }
    valid_metrics = {k: v for k, v in metric_options.items() if v and v in df.columns}
    if not valid_metrics:
        st.error("No mappable metrics found in dataset.")
        return

    metric_label = st.selectbox("Select Metric", list(valid_metrics.keys()))
    metric_col = valid_metrics[metric_label]

    # Year selection / animation
    years = sorted(df["year"].unique())
    if len(years) > 1:
        use_animation = st.checkbox("Animate over years", value=False)
        if not use_animation:
            selected_year = st.select_slider("Year", options=years, value=years[-1])
            plot_df = df[df["year"] == selected_year]
        else:
            plot_df = df  # will use animation_frame
    else:
        plot_df = df
        selected_year = years[0]

    # Continent filter
    continents = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    selected_continents = st.multiselect("Filter by Continent", continents, default=continents)
    continent_map = {
        "Africa": ["Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cabo Verde","Cameroon","Central African Republic","Chad","Comoros","Congo","DR Congo","Djibouti","Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Sao Tome and Principe","Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan","Sudan","Swaziland","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"],
        "Asia": ["Afghanistan","Armenia","Azerbaijan","Bahrain","Bangladesh","Bhutan","Brunei","Cambodia","China","Cyprus","Georgia","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Kazakhstan","Kuwait","Kyrgyzstan","Laos","Lebanon","Malaysia","Maldives","Mongolia","Myanmar","Nepal","North Korea","Oman","Pakistan","Philippines","Qatar","Saudi Arabia","Singapore","South Korea","Sri Lanka","Syria","Tajikistan","Thailand","Timor-Leste","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan","Vietnam","Yemen"],
        "Europe": ["Albania","Andorra","Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg","Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia","Norway","Poland","Portugal","Romania","Russia","San Marino","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Vatican City"],
        "North America": ["Antigua and Barbuda","Bahamas","Barbados","Belize","Canada","Costa Rica","Cuba","Dominica","Dominican Republic","El Salvador","Grenada","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Trinidad and Tobago","United States"],
        "South America": ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"],
        "Oceania": ["Australia","Fiji","Kiribati","Marshall Islands","Micronesia","Nauru","New Zealand","Palau","Papua New Guinea","Samoa","Solomon Islands","Tonga","Tuvalu","Vanuatu"]
    }
    allowed_countries = [c for cont in selected_continents for c in continent_map.get(cont, [])]
    plot_df = plot_df[plot_df["country"].isin(allowed_countries)]

    if plot_df.empty:
        st.info("No countries match the filter.")
        return

    # Build choropleth
    if len(years) > 1 and use_animation:
        fig = px.choropleth(
            plot_df,
            locations="country",
            locationmode="country names",
            color=metric_col,
            animation_frame="year",
            color_continuous_scale="viridis",
            title=f"{metric_label} Heatmap (animated)",
            hover_name="country",
            hover_data={metric_col: ":.2f"}
        )
    else:
        fig = px.choropleth(
            plot_df,
            locations="country",
            locationmode="country names",
            color=metric_col,
            color_continuous_scale="viridis",
            title=f"{metric_label} Heatmap ({selected_year if 'selected_year' in locals() else years[0]})",
            hover_name="country",
            hover_data={metric_col: ":.2f"}
        )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
    st.plotly_chart(fig, use_container_width=True)

    # Ranking table
    st.subheader("🏅 Country Ranking")
    latest = df.sort_values("year").groupby("country").last().reset_index()
    rank_df = latest[["country", metric_col]].dropna().sort_values(metric_col, ascending=False)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    csv = rank_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Ranking", csv, f"ranking_{metric_col}.csv", "text/csv")