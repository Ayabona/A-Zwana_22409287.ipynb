"""Johannesburg election explorer. Run with: streamlit run app.py."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from election_data import CASE_WARDS, PARTIES, YEARS, aggregate_history, load_data, ward_label
from election_model import SHARES, build_forecasts, leader, metro_projection

ROOT = Path(__file__).resolve().parent
COLORS = {"ANC": "#C59217", "DA": "#247BC1", "EFF": "#C84952", "ActionSA": "#19865B",
          "IFP": "#714B9F", "PA": "#BA5F21", "VF+": "#486549", "Other": "#8591A1"}
st.set_page_config(page_title="Johannesburg Election Explorer", page_icon="🗳️", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1420px; padding-top: 4rem;}
[data-testid="stMetric"] {background: white; border: 1px solid #DFE5ED;
border-radius: 12px; padding: 18px;}
h1 {letter-spacing: -.04em;} h2 {letter-spacing: -.025em;}
[data-testid="stSidebar"] {border-right: 1px solid #DFE5ED;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Reading the historical election files…")
def cached_data(signature):
    return load_data(ROOT)


@st.cache_data(show_spinner="Training the models and checking the 2021 backtest…")
def cached_forecasts(history):
    return build_forecasts(history)


def chart(fig, key):
    fig.update_layout(margin=dict(l=12, r=12, t=20, b=16), height=350,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#172B4D"), legend_title_text="",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "scrollZoom": False})


def share_long(frame, parties, value_name="Vote share (%)"):
    data = frame.melt(id_vars="Year", value_vars=[f"{p}_share" for p in parties],
                      var_name="Party", value_name=value_name)
    data["Party"] = data.Party.str.removesuffix("_share")
    data["Year"] = data.Year.astype(str)
    return data


def download(frame, name, label):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv")


def historical_view(history):
    st.caption("HISTORICAL OBSERVATIONS · 2011, 2016 AND 2021")
    st.title("What happened in Johannesburg?")
    st.write("Explore recorded election results. Vote shares and turnout rates are calculated from the supplied counts.")
    with st.sidebar:
        st.subheader("Historical filters")
        ballot = st.selectbox("Ballot", ["PR", "Ward"],
                              help="PR measures party-list votes. Ward ballots measure ward contests. They are never added together.")
        start, end = st.select_slider("Election years", options=list(YEARS), value=(2011, 2021))
        scope = st.radio("Geographic scope", ["Whole municipality", "Selected wards"])
        wards = sorted(history.Ward.unique())
        selected = st.multiselect("Wards", wards, default=list(CASE_WARDS), format_func=ward_label,
                                  disabled=scope == "Whole municipality", key="history_wards")
        parties = st.multiselect("Parties to display", PARTIES, default=list(PARTIES), key="history_parties")
        st.caption("Party filters affect party charts and exported party columns. Turnout and vote-share denominators retain all parties.")
    frame = history.loc[(history.BallotType == ballot) & history.Year.between(start, end)].copy()
    if scope == "Selected wards":
        if not selected:
            st.info("Select at least one ward to view its historical results.")
            return
        frame = frame.loc[frame.Ward.isin(selected)]
        available = set(frame.Ward)
        if set(selected) - available:
            st.warning("Some selected wards have no records in this period. No missing results have been filled with zeros.")
    if frame.empty:
        st.info("No observations match these filters. Select a different election or ward.")
        return
    summary = aggregate_history(frame)
    latest = summary.iloc[-1]
    latest_wards = frame.loc[frame.Year == latest.Year]
    st.caption(f"{scope} · {ballot} ballots · {start}–{end} · Latest observed election: {int(latest.Year)}")
    columns = st.columns(4)
    for col, label, value in zip(columns, ["Wards represented", "Valid ballots", "Reported voters", "Turnout"],
                                  [f"{len(latest_wards):,}", f"{latest.ValidVotes:,.0f}",
                                   f"{latest.Voters:,.0f}", f"{latest['Turnout (%)']:.1f}%"]):
        col.metric(label, value)
    st.caption("Turnout = reported voters ÷ (registered voters + MEC7 votes). Source reports estimate voters using the higher Ward/PR ballots cast; voters and valid ballots are different measures.")
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Party support over time")
        if parties:
            fig = px.line(share_long(summary, parties), x="Year", y="Vote share (%)", color="Party",
                          markers=True, color_discrete_map=COLORS)
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            fig.update_xaxes(type="category")
            chart(fig, "history_share")
        else:
            st.info("Select at least one party to display party support.")
    with right:
        st.subheader("Voter participation")
        fig = px.line(summary.assign(Year=summary.Year.astype(str)), x="Year", y="Turnout (%)", markers=True)
        fig.update_traces(line_color="#087E8B")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_xaxes(type="category")
        chart(fig, "history_turnout")
    if len(summary) > 1:
        change = latest['Turnout (%)'] - summary.iloc[0]['Turnout (%)']
        st.write(f"Across the selected area, turnout changed by **{change:+.1f} percentage points** "
                 f"between {int(summary.iloc[0].Year)} and {int(latest.Year)}. "
                 "These are weighted rates calculated from total voters and the total electorate.")
    st.info("Ward results use each election’s recorded boundaries. Matching ward numbers do not establish geographic comparability. See Model & data for voting-district changes.")
    with st.expander("Explore ward variation and descriptive statistics"):
        fig = px.scatter(latest_wards.assign(Ward=latest_wards.Ward.map(ward_label)),
                         x="RegisteredVoters", y="Turnout (%)", hover_name="Ward",
                         labels={"RegisteredVoters": "Registered voters"})
        chart(fig, "ward_variation")
        stats = latest_wards[["RegisteredVoters", "Voters", "Turnout (%)"]].describe().T
        st.dataframe(stats.round(2), width="stretch")
        st.caption("Summary statistics describe wards in the latest selected election; the mean ward rate differs from the weighted metro rate.")
    with st.expander("View and download the filtered historical data"):
        export = frame[["Year", "Ward", "BallotType", "RegisteredVoters", "MEC7", "Electorate", "Voters",
                        "Turnout (%)", "ValidVotes", *parties, *[f"{p}_share" for p in parties]]].copy()
        export["Record type"] = "Historical observation / derived rate"
        st.dataframe(export, hide_index=True, width="stretch")
        st.caption("Party count columns contain valid votes. Columns ending in _share contain proportions from 0 to 1.")
        download(export, "historical_filtered.csv", "Download historical data")


def prediction_view(history, audit):
    forecasts, metrics, matrices, errors = cached_forecasts(history)
    pr = forecasts.loc[forecasts.BallotType == "PR"].copy()
    wards = forecasts.loc[forecasts.BallotType == "Ward"].copy()
    metro = metro_projection(pr)
    metro_leader, top_share, _ = leader(metro.set_index("Party")["Projected share (%)"] / 100)
    st.caption("MODEL ESTIMATES · 2026 · JOHANNESBURG")
    st.title("Four questions for 2026")
    st.write("Projections for 04 November 2026, the scenario date specified in the supplied brief. Historical results end in 2021.")
    st.warning("These are conditional model estimates. Only three election cycles are available, and ward boundaries have not been spatially reconciled. No calibrated election probabilities are available.")
    with st.sidebar:
        st.subheader("Ward case studies")
        chosen = st.multiselect("Select three wards", sorted(wards.Ward), default=list(CASE_WARDS),
                                max_selections=3, format_func=ward_label, key="forecast_wards")
        st.caption("These controls affect question 2 only. Questions 1, 3 and 4 always cover the whole municipality.")
        st.caption("Default wards 1, 50 and 100 are retained from the supplied notebook. Their observed profiles are shown below; demographic claims have not been verified.")
    st.subheader("1. Which party leads the metro, and what about coalitions?")
    a, b = st.columns([1, 2])
    with a:
        st.metric("PR share leader", metro_leader)
        st.metric("Leading projected share", f"{100 * top_share:.1f}%")
    with b:
        named = metro.loc[metro.Party != "Other"]
        if named['Projected share (%)'].max() <= 50:
            st.write("**No individually modelled party exceeds 50% of projected PR votes.** "
                     "This is compatible with a fragmented result, but it does not establish whether a coalition will form.")
        else:
            st.write("**An individually modelled party exceeds 50% of projected PR votes.** "
                     "A vote-share majority alone does not establish a council-seat majority.")
        st.write("Coalition formation and who will govern cannot be estimated from this model. "
                 "It does not model council seat allocation, negotiations or coalition agreements.")
        st.caption("The metro comparison uses PR ballots as a consistent party-list measure. It is not a combined council-seat forecast.")
    st.divider()
    st.subheader("2. Which party leads in the three selected wards?")
    if len(chosen) != 3:
        st.info("Select exactly three wards in the sidebar to complete the three-ward comparison.")
    for column, ward in zip(st.columns(3), chosen):
        row = wards.loc[wards.Ward == ward].iloc[0]
        name, share, gap = leader(row[SHARES].to_numpy())
        observed = history.loc[(history.Ward == ward) & (history.BallotType == "Ward") & (history.Year == 2021)].iloc[0]
        old_leader, old_share, old_gap = leader(observed[SHARES].to_numpy())
        geography = audit.loc[audit.Ward == ward].iloc[0]
        with column:
            with st.container(border=True):
                st.markdown(f"#### {ward_label(ward)}")
                st.metric("Projected leader", name)
                st.write(f"Projected share: **{share:.1%}**. Gap to the next category: **{gap * 100:.1f} pp**.")
                if gap < 0.05:
                    st.warning("Close estimate: less than 5 percentage points separate the two leading categories.")
                st.caption(f"Observed 2021: {old_leader}, {old_share:.1%}; lead of {old_gap * 100:.1f} pp; turnout {observed['Turnout (%)']:.1f}%.")
                st.caption(f"Geography check: {int(geography['Districts common to all years'])} of {int(geography['2021 districts'])} 2021 district IDs appear in this ward in all three years.")
                st.caption("Applies to the recorded ward-ID series. A verified forecast for the 2026 ward geography is unavailable without a boundary crosswalk.")
    st.caption("Case-study rationale: compare the notebook’s three named wards using their measured party margins and participation. No unverified descriptions such as ‘swing ward’ or demographic change are assumed. Other is a pooled category and cannot be declared a party winner.")
    st.divider()
    st.subheader("3. What vote total and share is projected for each party?")
    chart(px.bar(metro, x="Party", y="Projected share (%)", color="Party", color_discrete_map=COLORS,
                 text=metro['Projected share (%)'].map(lambda x: f"{x:.1f}%")), "forecast_share")
    table = metro.copy()
    table["2021 backtest ward RMSE (pp)"] = [errors['PR'][f"{p}_share"] for p in table.Party]
    table["Sensitivity low (%)"] = (table['Projected share (%)'] - table['2021 backtest ward RMSE (pp)']).clip(0, 100)
    table["Sensitivity high (%)"] = (table['Projected share (%)'] + table['2021 backtest ward RMSE (pp)']).clip(0, 100)
    display = table.rename(columns={"Projected votes": "Votes", "Projected share (%)": "Share (%)",
                                    "2021 backtest ward RMSE (pp)": "Backtest RMSE (pp)"}).copy()
    display["Sensitivity (%)"] = [f"{lo:.1f}–{hi:.1f}" for lo, hi in
                                  zip(table['Sensitivity low (%)'], table['Sensitivity high (%)'])]
    display = display[["Party", "Votes", "Share (%)", "Backtest RMSE (pp)", "Sensitivity (%)"]]
    st.dataframe(display.style.format({"Votes": "{:,.0f}", "Share (%)": "{:.1f}", "Backtest RMSE (pp)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption("Sensitivity limits apply ± one historical ward RMSE to the metro share as a stress test. They are not confidence intervals or probability statements, and their bounds need not sum to 100%.")
    st.caption("All party shares, including Other, sum to 100%. Other combines remaining parties and independents. ActionSA has only one observed election in these files; its forecast is especially uncertain.")
    st.divider()
    st.subheader("4. How many voters are projected to participate, and at what rate?")
    electorate, voters = pr.Electorate.sum(), pr.Voters.sum()
    rate = 100 * voters / electorate
    rmse = errors['PR']['Turnout (%)']
    low, high = max(0, rate - rmse), min(100, rate + rmse)
    a, b, c = st.columns(3)
    a.metric("Projected voters", f"{voters:,.0f}")
    b.metric("Projected turnout", f"{rate:.1f}%")
    c.metric("Assumed electorate", f"{electorate:,.0f}")
    st.write(f"Historical-error sensitivity: **{low:.1f}%–{high:.1f}%**, equivalent to "
             f"**{electorate * low / 100:,.0f}–{electorate * high / 100:,.0f} voters** at the fixed electorate. "
             "This is not a 95% prediction interval.")
    st.caption("The electorate holds each ward’s 2021 registered voters plus MEC7 count constant. The actual 2026 electorate is unknown. "
               "Projected valid PR votes use each ward’s 2021 valid-PR-votes-to-reported-voters ratio; valid votes exclude spoilt ballots and can differ from reported voter turnout.")
    with st.expander("Download the model estimates"):
        export = forecasts.copy()
        export["Record type"] = "Conditional 2026 model estimate"
        export["Model"] = "Random Forest, one-election lag, 150 trees, seed 42"
        export["Assumption"] = "2021 electorate fixed; recorded ward IDs; boundaries not reconciled"
        download(export, "2026_conditional_ward_estimates.csv", "Download ward estimates")
        download(table, "2026_metro_PR_estimates.csv", "Download metro estimates")


def evidence_view(history, audit):
    _, metrics, matrices, _ = cached_forecasts(history)
    st.caption("METHOD, VALIDATION AND SOURCE DATA")
    st.title("How to read the estimates")
    st.write("The dashboard rebuilds its estimates from the six supplied files. It keeps the notebook’s Random Forest approach, "
             "with corrected turnout parsing, separate ballot types and a later-election backtest.")
    st.subheader("A test on a later election")
    st.write("Train on 2011 features → 2016 outcomes; evaluate on 2016 features → 2021 outcomes. "
             "After evaluation, train on both transitions and use 2021 features to estimate 2026. "
             "Features are the previous election’s party shares and turnout. No 2026 observations are used.")
    st.caption("Turnout uses only previous turnout, so its projected voter count is identical for PR and Ward ballots. Party-share models use both shares and turnout.")
    st.caption("There are 130 paired wards in the training transition and 135 in the test transition. The backtest excludes all 2021 targets during training. "
               "One held-out election cannot establish reliable long-term error bounds. Ward-ID joins remain geographically conditional.")
    ballot = st.selectbox("Validation ballot", ["PR", "Ward"], key="validation_ballot")
    st.dataframe(metrics.loc[metrics.Ballot == ballot].round(2), hide_index=True, width="stretch")
    st.caption("MAE = average absolute error; RMSE gives larger errors more weight. Both are percentage points. "
               "Ridge and last-election persistence provide comparisons; Random Forest is retained to match the notebook’s model family, not asserted to be best on every target.")
    result = matrices[ballot]
    a, b = st.columns(2)
    a.metric("2021 leader accuracy", f"{result['accuracy']:.1%}")
    b.metric("2021 leader macro F1", f"{result['macro_f1']:.3f}")
    st.write("Confusion matrix for the largest predicted share: rows are observed leader categories; columns are predicted categories.")
    st.dataframe(result['matrix'], width="stretch")
    st.caption("This evaluates the share-based leader rule used in the dashboard. It is not a separate classifier’s accuracy or the probability that a 2026 leader is correct.")
    st.subheader("Geographic comparability")
    st.write("There are 130 ward IDs in 2011 and 135 in 2016/2021. The table checks the voting-district IDs recorded under each ward. "
             "A shared ward or district ID does not prove unchanged boundaries. No polygon crosswalk or 2026 delimitation is supplied; "
             "ward-level findings are exploratory and conditional, not geographically verified predictions.")
    st.dataframe(audit, hide_index=True, width="stretch")
    download(audit, "ward_geography_audit.csv", "Download geography checks")
    st.subheader("Sources and definitions")
    st.write("The supplied exports identify Johannesburg and report IEC-style election returns. Original download URLs and acquisition dates "
             "are not present in the project; external provenance has not been independently verified.")
    st.dataframe(pd.DataFrame([{"Election": year, "Results file": f"jhb_{year}_detailed_results.csv",
                               "Turnout file": f"jhb_{year}_voter_turnout.xls",
                               "Wards": history.loc[(history.Year == year) & (history.BallotType == 'PR'), 'Ward'].nunique()}
                              for year in YEARS]), hide_index=True, width="stretch")
    st.markdown("""
- **Historical counts:** party votes, registered voters and reported voter turnout from the files.
- **Derived historical measures:** shares, rates, weighted aggregates and district overlap.
- **Predictions:** 2026 Random Forest estimates, conditional on fixed 2021 electorate and recorded ward IDs.
- **Ballots:** PR and Ward records remain separate. Repeated registration and spoilt-ballot metadata are counted once per district and ballot.
- **Turnout:** the report denominator is registered voters plus MEC7 votes. The reported voter count is a ballot-based turnout proxy, not a direct count of unique persons.
- **Coverage:** all 135 wards from 2021 feed metro forecasts. Seven named parties are modelled separately; remaining parties and independents stay in Other.
- **Limitations:** three cycles, changing geography, emerging parties, fixed electorate and no coalition/seat model. Do not treat error sensitivities as calibrated confidence intervals.
""")


st.sidebar.markdown("## Johannesburg\nElection Explorer")
page = st.sidebar.radio("View", ["Historical EDA · 2011–2021", "Predictions · 2026", "Model & data"], key="page")
st.sidebar.divider()
try:
    paths = [ROOT / f"jhb_{year}_{suffix}" for year in YEARS
             for suffix in ("detailed_results.csv", "voter_turnout.xls")]
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    history, audit = cached_data(signature)
except (OSError, ValueError, KeyError, ImportError) as exc:
    st.error(f"The dashboard could not load its source data: {exc}")
    st.info("Keep all six election files beside app.py and install the packages in requirements.txt.")
    st.stop()
if page.startswith("Historical"):
    historical_view(history)
elif page.startswith("Predictions"):
    prediction_view(history, audit)
else:
    evidence_view(history, audit)
st.divider()
st.caption("Johannesburg Election Explorer · Historical observations 2011–2021 · Conditional projections 2026")
