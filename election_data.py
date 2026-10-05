"""Read the supplied IEC exports without changing the source files."""
from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import xlrd

YEARS = (2011, 2016, 2021)
PARTIES = ("ANC", "DA", "EFF", "ActionSA", "IFP", "PA", "VF+", "Other")
CASE_WARDS = (79800001, 79800050, 79800100)
PARTY_NAMES = {
    "AFRICAN NATIONAL CONGRESS": "ANC",
    "DEMOCRATIC ALLIANCE/DEMOKRATIESE ALLIANSIE": "DA",
    "DEMOCRATIC ALLIANCE": "DA",
    "ECONOMIC FREEDOM FIGHTERS": "EFF",
    "ACTIONSA": "ActionSA",
    "INKATHA FREEDOM PARTY": "IFP",
    "PATRIOTIC ALLIANCE": "PA",
    "VRYHEIDSFRONT PLUS": "VF+",
    "FREEDOM FRONT PLUS": "VF+",
}


def ward_label(ward):
    return f"Ward {int(ward) % 1000} ({int(ward)})"


def read_results(path, year):
    path = Path(path)
    with path.open("rb") as handle:
        encoding = "utf-16" if handle.read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    df = pd.read_csv(path, encoding=encoding)
    required = {"Municipality", "Ward", "VotingDistrict", "BallotType", "PartyName",
                "TotalValidVotes", "RegisteredVoters", "SpoiltVotes"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path.name}: missing columns {required - set(df.columns)}")
    if not df.Municipality.str.startswith("JHB - City of Johannesburg").all():
        raise ValueError(f"{path.name}: expected Johannesburg records only")
    df["Ward"] = df.Ward.astype(str).str.extract(r"(\d{8})", expand=False).astype(int)
    for col in ("VotingDistrict", "TotalValidVotes", "RegisteredVoters", "SpoiltVotes"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any() or (df[col] < 0).any():
            raise ValueError(f"{path.name}: invalid {col}")
    if not df.BallotType.isin(["PR", "Ward"]).all():
        raise ValueError(f"{path.name}: unexpected ballot type")
    df["Party"] = df.PartyName.str.strip().str.upper().map(PARTY_NAMES).fillna("Other")
    df["Year"] = year
    return df


def aggregate_results(raw):
    """One row per year, ward and ballot; repeated station metadata counted once."""
    keys = ["Year", "Ward", "BallotType"]
    station_keys = keys + ["VotingDistrict"]
    variation = raw.groupby(station_keys)[["RegisteredVoters", "SpoiltVotes"]].nunique()
    if (variation > 1).any().any():
        raise ValueError("Conflicting voter/spoilt-ballot counts for the same voting district")
    stations = raw.drop_duplicates(station_keys)
    counts = stations.groupby(keys)[["RegisteredVoters", "SpoiltVotes"]].sum()
    votes = raw.groupby(keys + ["Party"]).TotalValidVotes.sum().unstack(fill_value=0)
    votes = votes.reindex(columns=PARTIES, fill_value=0)
    result = counts.join(votes)
    result["ValidVotes"] = result[list(PARTIES)].sum(axis=1)
    result["BallotsCast"] = result.ValidVotes + result.SpoiltVotes
    for party in PARTIES:
        result[f"{party}_share"] = result[party] / result.ValidVotes
    if (result.ValidVotes <= 0).any() or (result.RegisteredVoters <= 0).any():
        raise ValueError("A ward has no valid ballots or registered voters")
    return result.reset_index()


def aggregate_history(frame):
    """Recompute denominators before any party display filter is applied."""
    cols = ["RegisteredVoters", "Electorate", "Voters", "ValidVotes", "SpoiltVotes", *PARTIES]
    grouped = frame.groupby("Year")[cols].sum()
    grouped["Turnout (%)"] = 100 * grouped.Voters / grouped.Electorate
    for party in PARTIES:
        grouped[f"{party}_share"] = 100 * grouped[party] / grouped.ValidVotes
    return grouped.reset_index()


def geography_audit(raw):
    """Station-ID overlap is a diagnostic, never proof of unchanged boundaries."""
    rows = []
    station_sets = raw.groupby(["Year", "Ward"]).VotingDistrict.agg(set)
    for ward in sorted(raw.loc[raw.Year == 2021, "Ward"].unique()):
        latest = station_sets.get((2021, ward), set())
        sets = [station_sets.get((y, ward), set()) for y in YEARS]
        common = set.intersection(*sets)
        rows.append({"Ward": ward, "2011 districts": len(sets[0]),
                     "2016 districts": len(sets[1]), "2021 districts": len(latest),
                     "Districts common to all years": len(common),
                     "Same district IDs": sets[0] == sets[1] == sets[2],
                     "2021 districts retained (%)": 100 * len(common) / len(latest)})
    return pd.DataFrame(rows)


def read_turnout(path, year):
    book = xlrd.open_workbook(str(path), logfile=io.StringIO())
    sheet = book.sheet_by_index(0)
    rows = [sheet.row_values(i) for i in range(sheet.nrows)]
    for index, row in enumerate(rows):
        labels = [re.sub(r"\s+", " ", str(value)).strip().lower() for value in row]
        if "ward" in labels and any(x.startswith("registered voters") for x in labels):
            break
    else:
        raise ValueError(f"{Path(path).name}: turnout headers not found")
    def column(prefix):
        matches = [i for i, label in enumerate(labels) if label.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"{Path(path).name}: ambiguous {prefix} column")
        return matches[0]
    columns = {"Ward": column("ward"), "ReportedRegistered": column("registered voters"),
               "MEC7": column("mec7"), "Voters": column("voter turnout"),
               "ReportedRate": column("% voter turnout")}
    records = [{name: row[i] for name, i in columns.items()}
               for row in rows[index + 1:]
               if re.fullmatch(r"798\d{5}(?:\.0)?", str(row[columns["Ward"]]).strip())]
    result = pd.DataFrame(records).apply(pd.to_numeric, errors="raise")
    result["Ward"] = result.Ward.astype(int)
    result["Year"] = year
    result["Electorate"] = result.ReportedRegistered + result.MEC7
    if result.empty or result.Ward.duplicated().any():
        raise ValueError(f"{Path(path).name}: missing or duplicated wards")
    if (result.Electorate <= 0).any() or not result.Voters.between(0, result.Electorate).all():
        raise ValueError(f"{Path(path).name}: invalid voter counts")
    if not np.allclose(result.Voters / result.Electorate, result.ReportedRate, atol=1e-7):
        raise ValueError(f"{Path(path).name}: turnout does not reconcile to the reported rate")
    return result


def load_data(directory):
    directory = Path(directory)
    raw = pd.concat([read_results(directory / f"jhb_{y}_detailed_results.csv", y)
                     for y in YEARS], ignore_index=True)
    history = aggregate_results(raw)
    turnout = pd.concat([read_turnout(directory / f"jhb_{y}_voter_turnout.xls", y)
                         for y in YEARS], ignore_index=True)
    history = history.merge(turnout, on=["Year", "Ward"], how="left", validate="many_to_one")
    if history[["Voters", "ReportedRegistered"]].isna().any().any():
        raise ValueError("The turnout reports do not cover every ward in the results")
    if not np.array_equal(history.RegisteredVoters, history.ReportedRegistered):
        raise ValueError("Turnout and results files disagree on registered voters")
    history["Turnout (%)"] = 100 * history.Voters / history.Electorate
    return history, geography_audit(raw)
