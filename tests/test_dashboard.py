"""Source reconciliation, forecast coherence and examiner interaction tests."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from election_data import PARTIES, aggregate_history, load_data
from election_model import SHARES, build_forecasts, leader, metro_projection, transition


@pytest.fixture(scope="module")
def data():
    return load_data(ROOT)


@pytest.fixture(scope="module")
def model(data):
    return build_forecasts(data[0])


def test_source_totals_and_turnout_denominator(data):
    history, _ = data
    actual = aggregate_history(history.loc[history.BallotType == "PR"]).set_index("Year")
    assert actual.loc[2021, "RegisteredVoters"] == 2220710
    assert actual.loc[2021, "Voters"] == 947305
    assert actual.loc[2021, "ValidVotes"] == 923724
    assert actual.loc[2021, "Electorate"] == 2223297
    assert actual.loc[2021, "Turnout (%)"] == pytest.approx(100 * 947305 / 2223297)
    assert actual.loc[2011, "RegisteredVoters"] == 2010121
    assert np.allclose(history[SHARES].sum(axis=1), 1)
    assert history.groupby(["Year", "BallotType"]).size().tolist() == [130, 130, 135, 135, 135, 135]


def test_ballots_not_added_and_missing_years_not_fabricated(data):
    history, audit = data
    assert history.loc[(history.Year == 2011) & (history.Ward == 79800135)].empty
    assert not audit.loc[audit.Ward.isin([79800001, 79800050, 79800100]), "Same district IDs"].any()
    with pytest.raises(ValueError, match="one ballot"):
        metro_projection(history)


def test_forecast_all_wards_shares_and_voters_reconcile(model, data):
    forecast, metrics, _, _ = model
    assert forecast.groupby("BallotType").size().tolist() == [135, 135]
    assert np.isfinite(forecast[SHARES + ["Voters", "ValidVotes"]]).all().all()
    assert np.allclose(forecast[SHARES].sum(axis=1), 1)
    assert forecast["Turnout (%)"].between(0, 100).all()
    assert np.allclose(forecast.Voters, forecast.Electorate * forecast["Turnout (%)"] / 100)
    assert np.allclose(forecast[list(PARTIES)].sum(axis=1), forecast.ValidVotes)
    assert (forecast.ValidVotes <= forecast.Voters).all()
    assert np.allclose(forecast.loc[forecast.BallotType == "PR", "Voters"],
                       forecast.loc[forecast.BallotType == "Ward", "Voters"])
    for ballot in ["PR", "Ward"]:
        subtotal = metro_projection(forecast.loc[forecast.BallotType == ballot])
        assert subtotal['Projected share (%)'].sum() == pytest.approx(100)
    assert set(metrics.Model) == {"Random Forest", "Ridge", "Last election"}
    assert (metrics['Test wards'] == 135).all()
    assert len(transition(data[0].loc[data[0].BallotType == "PR"], 2011, 2016)[0]) == 130
    assert leader([0.1, 0.1, 0, 0, 0, 0, 0, 0.8])[0] == "Unresolved (Other leads)"
    assert leader([0.5, 0.5, 0, 0, 0, 0, 0, 0])[0] == "Unresolved (tie)"


def test_app_filters_forecasts_and_evidence():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert app.metric[1].value == "923,724"
    app.multiselect(key="history_parties").set_value(["DA"]).run()
    assert not app.exception
    assert app.metric[1].value == "923,724"  # Display filters cannot inflate shares.
    app.multiselect(key="history_parties").set_value([]).run()
    assert any("at least one party" in item.value for item in app.info)
    app.radio[1].set_value("Selected wards").run()
    app.multiselect(key="history_wards").set_value([]).run()
    assert any("at least one ward" in item.value for item in app.info)
    app.multiselect(key="history_wards").set_value([79800135]).run()
    app.select_slider[0].set_value((2011, 2011)).run()
    assert any("No observations" in item.value for item in app.info)
    app.radio(key="page").set_value("Predictions · 2026").run()
    assert not app.exception
    questions = [item.value for item in app.subheader]
    assert all(any(text.startswith(f"{n}.") for text in questions) for n in [1, 2, 3, 4])
    before = [item.value for item in app.metric]
    app.multiselect(key="forecast_wards").set_value([79800001]).run()
    assert not app.exception
    assert [item.value for item in app.metric][-3:] == before[-3:]  # Metro turnout unaffected.
    assert any("exactly three" in item.value for item in app.info)
    app.radio(key="page").set_value("Model & data").run()
    assert not app.exception
    app.selectbox(key="validation_ballot").set_value("Ward").run()
    assert not app.exception
