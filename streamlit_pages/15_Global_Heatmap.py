# streamlit_pages/15_Global_Heatmap.py
"""Global Heatmap with Plotly."""
import streamlit as st
import plotly.express as px
import pandas as pd

def render(df: pd.DataFrame):
    st.title("🌡️ Global Heatmap")
    st.markdown("Interactive world map of EV metrics.")

    if df.empty:
        st.warning("No data.")
        return

    # Aggregate latest year per country
    latest = df.sort_values("year").groupby("country").last().reset_index()

    metric_options = {
        "EVQTI": "evqti" if "evqti" in latest.columns else None,
        "ETRI": "ETRI" if "ETRI" in latest.columns else None,
        "Penetration": "penetration",
        "Growth": "growth_rate",
        "Infrastructure": "charging_points_per_100k",
        "Sales": "ev_sales",
        "Stock": "ev_stock",
    }
    valid_metrics = {k: v for k, v in metric_options.items() if v and v in latest.columns}
    if not valid_metrics:
        st.error("No mappable metrics found.")
        return

    metric = st.selectbox("Select Metric", list(valid_metrics.keys()))
    col = valid_metrics[metric]

    # Continent filter
    continent_map = {
        'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon',
                   'Central African Republic', 'Chad', 'Comoros', 'Congo', 'DR Congo', 'Djibouti', 'Egypt',
                   'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
                   'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
                   'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
                   'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
                   'South Sudan', 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
        'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia',
                'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan',
                'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia',
                'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Philippines', 'Qatar', 'Saudi Arabia',
                'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste',
                'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
        'Europe': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria',
                  'Croatia', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
                  'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania',
                  'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
                  'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia',
                  'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'],
        'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba',
                         'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti',
                         'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis',
                         'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
        'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
                         'Peru', 'Suriname', 'Uruguay', 'Venezuela'],
        'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand',
                   'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']
    }
    all_continents = list(continent_map.keys())
    selected_continents = st.multiselect("Continents", all_continents, default=all_continents)
    countries_to_show = []
    for cont in selected_continents:
        countries_to_show.extend(continent_map[cont])
    map_df = latest[latest["country"].isin(countries_to_show)]

    if map_df.empty:
        st.info("No countries selected.")
        return

    fig = px.choropleth(map_df, locations="country", locationmode="country names",
                        color=col, color_continuous_scale="viridis",
                        title=f"{metric} Heatmap", hover_name="country",
                        hover_data={col: ":.2f"})
    st.plotly_chart(fig, use_container_width=True)

    # Ranking table
    st.subheader("Ranking")
    st.dataframe(map_df[["country", col]].sort_values(col, ascending=False).head(20))