import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from dotenv import load_dotenv
from pathlib import Path

# Try importing LangChain for the chatbot capability
try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI
    has_langchain = True
except ImportError:
    has_langchain = False

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

st.set_page_config(page_title="Denominator Assessment", page_icon="📊", layout="wide")

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"], * { font-family: 'DM Sans', sans-serif !important; }

    .stApp { background: #f7f8fc; }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f2d5c 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 250px; height: 260px;
        background: radial-gradient(circle, rgba(99,110,250,0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .dashboard-header::after {
        content: '';
        position: absolute;
        bottom: -80px; left: 40%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 65%);
        border-radius: 50%;
    }
    .title-main {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.8rem; font-weight: 800; color: #ffffff;
        letter-spacing: -0.02em; margin: 0 0 6px 0; line-height: 1.1;
    }
    .title-accent { color: #38bdf8; }
    .subtitle-text {
        color: #94a3b8; font-size: 1.05rem; font-weight: 400;
        margin: 0; max-width: 700px; line-height: 1.6;
    }
    .subtitle-text b { color: #cbd5e1; font-weight: 600; }

    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e8edf5;
        padding: 26px 22px 22px; border-radius: 16px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative; overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        border-radius: 16px 16px 0 0;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(37,99,235,0.13);
        border-color: #bfdbfe;
    }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

    h3, .stSubheader { font-family: 'Syne', sans-serif !important; color: #0f172a !important; font-weight: 700 !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #eef2f9; border-radius: 12px; padding: 4px; border: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 22px; font-weight: 600;
        color: #64748b; background: transparent; border: none; transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important; color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.12);
    }

    .stSelectbox label { font-weight: 600; color: #374151; }

    hr { border: none; border-top: 1px solid #e8edf5; margin: 1.5rem 0; }

    .chat-header {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd; border-radius: 16px;
        padding: 20px 28px; margin-bottom: 1rem;
    }
    .chat-header h3 { margin: 0; color: #0369a1 !important; font-size: 1.15rem !important; }

    [data-testid="stCheckbox"] { display: none !important; }

    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #e8edf5; }

    .metric-section-label {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── File paths ────────────────────────────────────────────────────────────────
ENUMERATION_FILE = "enumeration_kaduna_bauchi_adamawa_gombe_yobe.csv"
IE_FILE          = "ie_jigawa_katsina_kebbi_zamfara.csv"
MDA_FILE         = "mda_round_1.csv"
WORLD_POP_FILE   = "world_pop_2026.csv"
IMMUNISATION     = "immunisation.csv"

SOURCE_COLORS = {
    "Enumeration":                 "#636EFA",
    "IE (Identify and Enumerate)": "#00CC96",
    "MDA Round":                   "#FFA15A",
    "World Pop":                   "#AD0B7C",
    "Immunisation":                "#103E18",   # fixed casing
}

def normalize_lga(s):
    """Normalize LGA name: uppercase, collapse spaces, unify dashes."""
    s = str(s).strip().upper()
    s = re.sub(r'[-\u2013\u2014]+', '-', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*-\s*', '-', s)
    return s

def load_data():
    df_enum          = pd.read_csv(ENUMERATION_FILE)
    df_ie            = pd.read_csv(IE_FILE)
    df_mda           = pd.read_csv(MDA_FILE)
    df_world_pop     = pd.read_csv(WORLD_POP_FILE)
    df_immunisation  = pd.read_csv(IMMUNISATION)

    # ── Add source labels ─────────────────────────────────────────────────────
    df_enum['source_type']          = 'Enumeration'
    df_ie['source_type']            = 'IE (Identify and Enumerate)'
    df_mda['source_type']           = 'MDA Round'
    df_world_pop['source_type']     = 'World Pop'
    df_immunisation['source_type']  = 'Immunisation'

    # ── Normalize columns ─────────────────────────────────────────────────────
    for df in [df_enum, df_ie, df_mda, df_world_pop, df_immunisation]:
        if 'state' in df.columns:
            df['state'] = df['state'].str.title()
        if 'local_government_area' in df.columns:
            df['local_government_area'] = df['local_government_area'].apply(normalize_lga)

    # ── Combine all datasets ──────────────────────────────────────────────────
    df_combined = pd.concat(
        [df_enum, df_ie, df_mda, df_world_pop, df_immunisation],
        ignore_index=True
    )

    return df_enum, df_ie, df_mda, df_world_pop, df_immunisation, df_combined

try:
    df_enum, df_ie, df_mda, df_world_pop, df_immunisation, df_combined = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
  <p class="title-main">📊 Baseline <span class="title-accent">Comparison</span> Portal</p>
  <p class="subtitle-text">
    A unified, intelligent view of population targets <b>(1–59 months)</b> across
    <b>Enumeration</b>, <b>IE (Identify & Enumerate)</b>, <b>MDA Round</b>, and <b>World Pop</b> datasets
    spanning Nigeria's northern states.
  </p>
</div>
""", unsafe_allow_html=True)

# ── CHATBOT ───────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="chat-header"><h3>💬 Ask Your Data Questions</h3></div>', unsafe_allow_html=True)

    #grok_key = os.getenv("GROK_API_KEY")
    grok_key = st.secrets["GROK_API_KEY"]

    if not grok_key:
        st.warning("⚠️ Chatbot disabled — add your GROK_API_KEY to the .env file.")
    elif not has_langchain:
        st.error("LangChain packages not found. Run: `pip install langchain langchain-experimental langchain-openai`")
    else:
        try:
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "Hello! I can help you explore the datasets. Try asking: *Which LGA has the highest population target in Kaduna?*"}
                ]

            for msg in st.session_state.messages:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

            if prompt := st.chat_input("Ask a question about the data…"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user", avatar="👤").write(prompt)

                llm = ChatOpenAI(
                    model="llama-3.3-70b-versatile",
                    api_key=grok_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0,
                )
                agent = create_pandas_dataframe_agent(
                    llm,
                    [df_enum, df_ie, df_mda, df_world_pop, df_immunisation],
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    prefix=(
                        "You are analyzing five pandas dataframes. "
                        "df1 is Enumeration data, df2 is IE data, df3 is MDA Round, "
                        "df4 is World Pop, df5 is Immunisation. "
                        "They share similar columns."
                    ),
                )

                with st.spinner("Thinking…"):
                    try:
                        response_obj  = agent.invoke({"input": prompt})
                        response_text = response_obj.get("output", str(response_obj))
                    except Exception as eval_err:
                        err_str = str(eval_err)
                        if "Could not parse LLM output:" in err_str:
                            ans = err_str.split("Could not parse LLM output:")[1].split("For troubleshooting")[0].strip()
                            response_text = ans.strip("`").strip()
                        else:
                            response_text = f"Sorry, there was an error: {eval_err}"

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.chat_message("assistant", avatar="🤖").write(response_text)

        except Exception as e:
            st.error(f"Error starting chat agent: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ── KEY METRICS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<p class="metric-section-label">Enumeration</p>', unsafe_allow_html=True)
    st.metric("States Covered", df_enum['state'].nunique())

with col2:
    st.markdown('<p class="metric-section-label">IE Dataset</p>', unsafe_allow_html=True)
    st.metric("States Covered", df_ie['state'].nunique())

with col3:
    st.markdown('<p class="metric-section-label">Current MDA Round</p>', unsafe_allow_html=True)
    st.metric("States Covered", df_mda['state'].nunique())

with col4:
    st.markdown('<p class="metric-section-label">World Pop</p>', unsafe_allow_html=True)
    st.metric("States Covered", df_world_pop['state'].nunique())

with col5:
    st.markdown('<p class="metric-section-label">Immunisation</p>', unsafe_allow_html=True)
    st.metric("States Covered", df_immunisation['state'].nunique())

st.divider()

# ── CHART LAYOUT DEFAULTS ─────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    font_family="DM Sans",
    title_font_family="Syne",
    title_font_size=20,
    title_font_color="#0f172a",
    yaxis=dict(tickformat=",", gridcolor="#eef2f9", zeroline=False),
    xaxis=dict(gridcolor="#eef2f9"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        title_font_weight="bold",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e8edf5",
        borderwidth=1,
        font_size=13,
    ),
    margin=dict(t=60, b=50, l=50, r=30),
    hoverlabel=dict(
        bgcolor="#0f172a", font_color="#f8fafc",
        font_family="DM Sans", bordercolor="#0f172a",
    ),
)

# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
st.subheader("📍 Regional Comparison: Population Target (1–59 Months)")

tab1, tab2 = st.tabs(["  State-Level View  ", "  Detailed LGA View  "])

with tab1:
    state_sum = (
        df_combined
        .groupby(['state', 'source_type'])['1_59m']
        .sum()
        .reset_index()
    )

    fig_state = px.bar(
        state_sum,
        x='state',
        y='1_59m',
        color='source_type',
        title="Total 1–59m Targets by State",
        labels={
            '1_59m': 'Target Population (1–59m)',
            'state': 'State',
            'source_type': 'Dataset'
        },
        barmode='group',
        color_discrete_map=SOURCE_COLORS,
        hover_data={'1_59m': ':,'},
        template='plotly_white',
    )

    # ── CORRECT LABELS ──
    for trace in fig_state.data:
        trace.text = [f"{v:,.0f}" for v in trace.y]
        trace.textposition = "inside"
        trace.textangle = 270
        trace.insidetextanchor = "middle"
        trace.textfont = dict(size=11, color="white")
        trace.opacity = 0.92

        # ✅ correct way to remove bar border
        trace.marker = dict(line=dict(width=0))

    # ── SAFE LAYOUT ──
    layout_copy = CHART_LAYOUT.copy()
    layout_copy.pop("legend", None)
    layout_copy.pop("margin", None)

    fig_state.update_layout(
        **layout_copy,
        legend=dict(
            title_text="Dataset",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.15,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#e8edf5",
            borderwidth=1,
        ),
        margin=dict(t=80, b=50, l=50, r=20),
    )

    st.plotly_chart(fig_state, use_container_width=True)
with tab2:
    selected_state = st.selectbox(
        "Select a State to view its LGA breakdown:",
        sorted(df_combined['state'].dropna().unique()),
    )

    lga_data = df_combined[df_combined['state'] == selected_state].copy()

    # Aggregate first (LGA names already normalized at load time)
    lga_data = (
        lga_data
        .groupby(['local_government_area', 'source_type'], as_index=False)['1_59m']
        .sum()
    )

    # Strict alphabetical order — applied uniformly across ALL sources
    lga_order = sorted(lga_data['local_government_area'].unique())

    lga_data['local_government_area'] = pd.Categorical(
        lga_data['local_government_area'],
        categories=lga_order,
        ordered=True,
    )
    lga_data = lga_data.sort_values('local_government_area')

    fig_lga = px.line(
        lga_data,
        x='local_government_area',
        y='1_59m',
        color='source_type',
        markers=True,
        title=f"LGA Comparison — {selected_state}",
        labels={
            '1_59m': 'Target Population (1–59m)',
            'local_government_area': 'Local Government Area',
            'source_type': 'Dataset',
        },
        color_discrete_map=SOURCE_COLORS,
        template='plotly_white',
        category_orders={"local_government_area": lga_order},
    )
    fig_lga.update_layout(**CHART_LAYOUT)
    fig_lga.update_traces(line_shape="linear", line=dict(width=2.5), marker=dict(size=7))
    st.plotly_chart(fig_lga, use_container_width=True)

st.divider()

# ── PRETTY FORMATTER ────────────────────────────────────────────────────────
def prettify_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()

    display.rename(columns={
        "local_government_area": "LGA",
        "state": "State"
    }, inplace=True, errors="ignore")

    # ✅ FORCE STATE TO UPPERCASE
    if "State" in display.columns:
        display["State"] = display["State"].astype(str).str.upper()

    # Format numbers with commas
    for col in display.columns:
        if col not in ["State", "LGA", "Round"]:
            display[col] = display[col].apply(
                lambda x: (
                "NA" if pd.isna(x)
                else f"{int(round(float(x))):,}"
            )
        )

    # ✅ FIX: Force Round to whole number
    if "Round" in display.columns:
        display["Round"] = display["Round"].apply(
            lambda x: int(float(x)) if pd.notna(x) and str(x).replace(".", "").isdigit() else 0
        )

    return display


# ── CENTER TABLE CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    table th, table td {
        text-align: center !important;
        vertical-align: middle !important;
    }

    thead tr th {
        font-weight: 700 !important;
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background:#ffffff; border:1px solid #e8edf5; border-radius:14px;
    padding:16px 22px; font-family:'DM Sans',sans-serif;
    font-weight:700; color:#0f172a; font-size:1.05rem;
    box-shadow:0 2px 8px rgba(15,23,42,0.04); margin-bottom:12px;
">
    View Unified Comparison Table
</div>
""", unsafe_allow_html=True)


# ── STATE FILTER ─────────────────────────────────────────────────────────────
all_states = sorted(df_combined["state"].dropna().unique())

selected_states_tbl = st.multiselect(
    "Select State(s):",
    options=all_states,
    default=[all_states[0]],
)


# ── BUILD TABLE FUNCTION ─────────────────────────────────────────────────────
def build_comparison_table(df):

    # Aggregate data
    tmp = (
        df.groupby(["state", "local_government_area", "source_type"])["1_59m"]
        .sum()
        .reset_index()
    )

    # Pivot datasets
    pivot = tmp.pivot_table(
        index=["state", "local_government_area"],
        columns="source_type",
        values="1_59m",
        aggfunc="sum"
    ).reset_index()

    pivot.rename(columns={
        "state": "State",
        "local_government_area": "LGA"
    }, inplace=True)

    # Ensure dataset columns exist
    expected_cols = [
        "Enumeration",
        "IE (Identify and Enumerate)",
        "World Pop",
        "MDA Round",
        "Immunisation"
    ]

    for col in expected_cols:
        if col not in pivot.columns:
            pivot[col] = 0

    # ── MDA ROUND META COLUMN ──
    if "round" in df.columns:
        mda_round = (
            df[df["source_type"] == "MDA Round"]
            .groupby(["state", "local_government_area"])["round"]
            .first()
            .reset_index()
        )

        pivot = pivot.merge(
            mda_round,
            left_on=["State", "LGA"],
            right_on=["state", "local_government_area"],
            how="left"
        )

        pivot.drop(columns=["state", "local_government_area"], inplace=True)
        pivot.rename(columns={"round": "Round"}, inplace=True)
    else:
        pivot["Round"] = "-"

    # Fill numeric values
    numeric_cols = [
        "Enumeration",
        "IE (Identify and Enumerate)",
        "World Pop",
        "MDA Round",
        "Immunisation"
    ]

    pivot[numeric_cols] = pivot[numeric_cols].astype("Float64")

    # ── STATE TOTAL ROW ───────────────────────────────────────────────────────
    state_totals = (
        pivot.groupby("State")[numeric_cols]
        .sum()
        .reset_index()
    )

    state_totals["LGA"] = "TOTAL"
    state_totals["Round"] = "-"

    # Combine
    final_df = pd.concat([pivot, state_totals], ignore_index=True)

    # Column order
    final_df = final_df[
        [
            "State",
            "LGA",
            "Enumeration",
            "IE (Identify and Enumerate)",
            "World Pop",
            "Immunisation",
            "MDA Round",
            "Round"
        ]
    ]

    # Sort (TOTAL always last)
    final_df["is_total"] = final_df["LGA"].eq("TOTAL")
    final_df = final_df.sort_values(["State", "is_total", "LGA"])
    final_df.drop(columns=["is_total"], inplace=True)

    return final_df


# ── STYLE TOTAL ROW ──────────────────────────────────────────────────────────
def style_total_row(df):
    def highlight(row):
        if row["LGA"] == "TOTAL":
            return [
                "background-color: #fff3cd; font-weight: bold; color: #7a5c00;"
            ] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight, axis=1)


# ── APPLY FILTER ─────────────────────────────────────────────────────────────
df_filtered = df_combined.copy()
df_filtered = df_filtered[df_filtered["state"].isin(selected_states_tbl)]

comparison_table = build_comparison_table(df_filtered)


# ── DISPLAY TABLE ────────────────────────────────────────────────────────────
show_data = st.checkbox("Show / Hide Data Table", value=True, label_visibility="collapsed")

if show_data:
    st.dataframe(
        style_total_row(prettify_df(comparison_table)),
        use_container_width=True,
        hide_index=True
    )

