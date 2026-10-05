"""Reproducible Random Forest projections and a genuinely later-election backtest.

One-election lags make a 2016 training target and unseen 2021 test possible.
Geography is joined by recorded ward ID, so forecasts remain conditional on
comparability of those units; the UI exposes the station-ID audit.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from election_data import PARTIES

SHARES = [f"{p}_share" for p in PARTIES]
TARGETS = [*SHARES, "Turnout (%)"]


def transition(history, early, late):
    x = history.loc[history.Year == early].set_index("Ward")
    y = history.loc[history.Year == late].set_index("Ward")
    wards = x.index.intersection(y.index).sort_values()
    return x.loc[wards, TARGETS], y.loc[wards, TARGETS]


def constrain(prediction):
    result = np.asarray(prediction, dtype=float).copy()
    shares = np.clip(result[:, :-1], 0, None)
    totals = shares.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("The model produced an empty vote-share distribution")
    result[:, :-1] = shares / totals
    result[:, -1] = np.clip(result[:, -1], 0, 100)
    return result


def fit_predict(x, y, future, method):
    estimates = []
    for target in TARGETS:
        model = (RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42)
                 if method == "Random Forest" else make_pipeline(StandardScaler(), Ridge(alpha=10)))
        # One shared turnout signal gives the same people count for both ballots.
        features = ["Turnout (%)"] if target == "Turnout (%)" else TARGETS
        model.fit(x[features], y[target])
        estimates.append(model.predict(future[features]))
    return constrain(np.column_stack(estimates))


def leader(shares):
    """Other pools unrelated parties and cannot be declared a winning party."""
    shares = pd.Series(shares, index=PARTIES, dtype=float)
    ranking = shares.sort_values(ascending=False)
    name = ranking.index[0]
    if name == "Other":
        return "Unresolved (Other leads)", float(ranking.iloc[0]), float(ranking.iloc[0] - ranking.iloc[1])
    if np.isclose(ranking.iloc[0], ranking.iloc[1]):
        return "Unresolved (tie)", float(ranking.iloc[0]), 0.0
    return name, float(ranking.iloc[0]), float(ranking.iloc[0] - ranking.iloc[1])


def build_forecasts(history):
    forecasts, metrics, matrices, residuals = [], [], {}, {}
    for ballot in ("PR", "Ward"):
        data = history.loc[history.BallotType == ballot]
        train_x, train_y = transition(data, 2011, 2016)
        test_x, test_y = transition(data, 2016, 2021)
        predictions = {}
        for method in ("Random Forest", "Ridge", "Last election"):
            pred = (constrain(test_x.to_numpy()) if method == "Last election"
                    else fit_predict(train_x, train_y, test_x, method))
            predictions[method] = pred
            errors = pred - test_y.to_numpy()
            errors[:, :-1] *= 100
            for j, target in enumerate(TARGETS):
                metrics.append({"Ballot": ballot, "Model": method,
                                "Target": target.replace("_share", " share").replace("Turnout (%)", "Turnout"),
                                "MAE (pp)": np.abs(errors[:, j]).mean(),
                                "RMSE (pp)": np.sqrt(np.mean(errors[:, j] ** 2)),
                                "Test wards": len(test_x)})
            if method == "Random Forest":
                residuals[ballot] = pd.Series(np.sqrt(np.mean(errors ** 2, axis=0)), index=TARGETS)
        # Score the same share-based leader rule that appears in the interface.
        actual = [leader(row)[0] for row in test_y[SHARES].to_numpy()]
        predicted = [leader(row)[0] for row in predictions["Random Forest"][:, :-1]]
        labels = sorted(set(actual) | set(predicted))
        matrices[ballot] = {"matrix": pd.DataFrame(confusion_matrix(actual, predicted, labels=labels),
                                                   index=labels, columns=labels),
                            "accuracy": accuracy_score(actual, predicted),
                            "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
                            "n": len(test_x)}
        # Backtest is finished before the final model sees 2021 target values.
        final_x = pd.concat([train_x, test_x], ignore_index=True)
        final_y = pd.concat([train_y, test_y], ignore_index=True)
        latest = data.loc[data.Year == 2021].sort_values("Ward").copy()
        projected = fit_predict(final_x, final_y, latest[TARGETS], "Random Forest")
        forecast = latest[["Ward", "BallotType", "RegisteredVoters", "MEC7", "Electorate"]].copy()
        forecast[TARGETS] = projected
        forecast["Voters"] = forecast.Electorate * forecast["Turnout (%)"] / 100
        # Keep ballot-specific valid-vote yield; turnout is the report's voter proxy.
        forecast["ValidVotes"] = forecast.Voters * (latest.ValidVotes / latest.Voters).to_numpy()
        for party in PARTIES:
            forecast[party] = forecast.ValidVotes * forecast[f"{party}_share"]
        forecast["Year"] = 2026
        forecasts.append(forecast)
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(metrics), matrices, residuals


def metro_projection(forecast):
    """Call with exactly one ballot type so people are never counted twice."""
    if forecast.BallotType.nunique() != 1:
        raise ValueError("Select one ballot type before aggregating forecasts")
    total = forecast[list(PARTIES)].sum()
    return pd.DataFrame({"Party": PARTIES, "Projected votes": total.reindex(PARTIES).to_numpy(),
                         "Projected share (%)": 100 * total.reindex(PARTIES).to_numpy() / total.sum()})
