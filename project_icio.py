#!/usr/bin/env python3
"""
Project the (Canada-split) ICIO from 2022 to 2030, year by year, with
three methods, and compare them.

    oecd      OECD bridge regressions with ARIMA errors on national-accounts
              proxies + RAS, in previous year's and current prices
              (OECD 2025, Trends in Global Value Chains, Annex A)
    arima     univariate ARIMA(p,1,q) on each margin's 1995-2022 history
    external  EIA AEO (US) and IEO (regions) sector output, energy
              intensity and energy prices, IMF WEO GDP, deflators, trade

Every year 2023-2030 is projected from the previous year's balanced
table and re-balanced with RAS (signed RAS from cswans46/baltic), first in
previous year's prices, then in current prices. The method is the same as
in cswans46/OECD_biojet (shared package icio_projection/).

Canadian data centres
---------------------
With --split-canada-dsut the base table is split as in the main model
(CAN_J62_63 -> _CLD, _DPH, _OTH). The history exists only for CAN_J62_63,
so all three subsectors start from the parent's projection; CLD and DPH
are then driven by dc_output_scenarios.csv (capacity x output per MW, 2022
USD): volume = 2022 output x parent growth to the first scenario year, plus
the scenario's additional output after it (--dc-driver scenario, default).
India is not split and follows its own projection.

Steps
-----
1. History panel (once; reads every year of the econ RData, or CSVs):
       python project_icio.py build-panel --out data/projection/panel ^
           --A-file ... --X-file ... --FD-file ...
       python project_icio.py build-panel --cp-dir D:/ICIO/csv --pyp-dir D:/ICIO/pyp
2. External projections (IMF WEO; EIA with EIA_API_KEY set):
       python project_icio.py fetch-external --out data/projection/external.csv
3. Projection, comparison and backtests:
       python project_icio.py run --panel data/projection/panel ^
           --external data/projection/external.csv --out output/projection ^
           [model arguments: --data-dir ... --A-file ... --split-canada-dsut ...]

Outputs (per method, under --out): <method>/<year>_{pyp,cp}.npz tables and
ICIOecon_{A,X,FD}_<year>.RData (current prices, for the main model with
--icio-year), comparison/*.csv, backtest/*.csv, comparison/*.png.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from icio_projection import evaluate as E
from icio_projection import external as X
from icio_projection import methods as M
from icio_projection import panel as P
from icio_projection import project as PR
from icio_projection.table import Table, country_of, read_wide

LOG = logging.getLogger("datacentre_icio.projection")
YEARS = list(range(2023, 2031))


# ---------------------------------------------------------------------
# Loading through the main model
# ---------------------------------------------------------------------

def _model():
    import global_datacentre_icio_model_v2 as model
    return model


def _model_args(argv: list[str]):
    model = _model()
    old = sys.argv
    try:
        sys.argv = ["model"] + argv
        return model.parse_args()
    finally:
        sys.argv = old


def load_year(model_args, year: int) -> Table:
    """A, X, FD of one year from the econ RData via the model's own loader."""
    model = _model()
    if not getattr(model.pyreadr.read_r, "_cached", False):
        cached = functools.lru_cache(maxsize=8)(model.pyreadr.read_r)
        cached._cached = True
        model.pyreadr.read_r = cached
    model_args.icio_year = year
    A, Xs, FD = model.load_icio(model_args)
    return Table.from_axfd(A, Xs, FD)


def build_panel_rdata(model_args, years=range(1995, 2023)) -> pd.DataFrame:
    rows = []
    for y in years:
        t = load_year(model_args, y)
        for var, s in P.margins(t).items():
            rows.append(pd.DataFrame({"year": y, "prices": "cp", "var": var,
                                      "key": s.index.astype(str), "value": s.to_numpy(float)}))
        LOG.info("panel: %s read from RData", y)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------
# Base table, split and data-centre driver
# ---------------------------------------------------------------------

def base_table(model_args, base_csv: Path | None):
    """2022 table (split if requested) and the parent map of split keys."""
    model = _model()
    if base_csv is not None:
        t = read_wide(base_csv)
        A, Xs, FD = t.to_axfd()
    else:
        model_args.icio_year = 2022
        A, Xs, FD = model.load_icio(model_args)
    plan = None
    if model_args.split_canada_dsut:
        A, Xs, FD, plan = model.apply_canada_split(model_args, A, Xs, FD)
    base = Table.from_axfd(A, Xs, FD)
    parent_of = {k: plan.parent_key for k in plan.sub_keys} if plan else {}
    return base, plan, parent_of


def dc_scenario_index(data_dir: Path, country: str, base: Table, plan, gx_parent: pd.Series,
                      years=YEARS) -> pd.DataFrame | None:
    """Volume growth index of CLD and DPH from the capacity scenario."""
    path = data_dir / "dc_output_scenarios.csv"
    if plan is None or not path.exists():
        return None
    sc = pd.read_csv(path)
    sc = sc[sc["country"].astype(str).str.strip().eq(country)].copy()
    if sc.empty:
        return None
    sc["dc_output_usd_mn"] = sc["dc_capacity_mw"] * sc["price_mn_MW_2022"]
    sc = sc.set_index("year")["dc_output_usd_mn"].sort_index()
    y0 = int(sc.index.min())
    dc_keys = [plan.cloud_key, plan.hosting_key]
    x0 = base.x()[dc_keys]
    total0 = float(x0.sum())
    path_vol = {}
    for y in years:
        g_par = float(gx_parent[min(y, y0)]) if min(y, y0) in gx_parent.index else 1.0
        level = total0 * g_par
        if y >= y0:
            level += float(sc.get(y, sc.iloc[-1]) - sc.loc[y0])
        path_vol[y] = level / total0 if total0 > 0 else 1.0
    idx = pd.DataFrame({k: pd.Series(path_vol) for k in dc_keys}).T
    return idx[years]


def apply_dc_driver(proj: M.Projected, idx: pd.DataFrame | None) -> M.Projected:
    if idx is None:
        return proj
    for k in idx.index:
        ratio = idx.loc[k] / proj.gx.loc[k]
        proj.gx.loc[k] = idx.loc[k]
        proj.gva.loc[k] = proj.gva.loc[k] * ratio
    proj.notes.append(f"Data-centre scenario drives {list(idx.index)}")
    return proj


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

def cmd_build_panel(a, rest):
    if a.cp_dir:
        pan = P.build_panel(a.cp_dir, a.pyp_dir, pattern=a.pattern)
    else:
        pan = build_panel_rdata(_model_args(rest))
        if a.pyp_dir:
            pyp = P.build_panel(a.pyp_dir, None, pattern=a.pattern)
            pyp["prices"] = "pyp"
            pan = pd.concat([pan, pyp], ignore_index=True)
    print(P.save_panel(pan, a.out))


def cmd_fetch_external(a, rest):
    print(X.fetch_all(a.out, ieo_route=a.ieo_route, aeo_route=a.aeo_route))


def cmd_run(a, rest):
    if a.base_csv is not None:
        for flag in ("--A-file", "--X-file", "--FD-file"):
            if flag not in rest:
                rest = rest + [flag, "unused-with-base-csv"]
    margs = _model_args(rest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pan = P.load_panel(a.panel)
    ext = X.External.load(a.external) if a.external and Path(a.external).exists() else None
    if ext is None:
        LOG.warning("No external projections file (%s): the oecd and external methods need it "
                    "(python project_icio.py fetch-external).", a.external)
        a.methods = [m for m in a.methods if m == "arima"]
    base, plan, parent_of = base_table(margs, a.base_csv)
    LOG.info("Base table: %s industries; split: %s", len(base.keys), bool(plan))

    chains, notes = {}, {}
    for m in a.methods:
        proj = M.project(m, pan, base, YEARS, ext=ext, parent_of=parent_of)
        if plan is not None and a.dc_driver == "scenario":
            idx = dc_scenario_index(Path(margs.data_dir), "CAN", base, plan, proj.gx.loc[plan.parent_key]
                                    if plan.parent_key in proj.gx.index else proj.gx.loc[plan.other_key])
            proj = apply_dc_driver(proj, idx)
        chains[m] = PR.run_chain(base, proj, out_dir=out if a.write_tables else None,
                                 write_rdata=a.write_rdata)
        chains[m].ras_log.to_csv(out / f"ras_log_{m}.csv", index=False)
        chains[m].margins.to_csv(out / f"margins_{m}.csv.gz", index=False)
        notes[m] = proj.notes

    focus = ["CAN_J62_63_CLD", "CAN_J62_63_DPH", "CAN_J62_63_OTH", "CAN_J62_63", "CAN_D35", "CAN_D",
             "IND_J62_63", "IND_D35", "IND_D", "USA_J62_63"]
    cmp_ = E.compare(chains, base, focus_keys=focus)
    E.write(cmp_, out / "comparison")
    if "focus_paths" in cmp_:
        try:
            E.plot_focus(cmp_["focus_paths"], out / "comparison" / "focus_paths.png",
                         "Gross output 2022-2030 by projection method (current prices)")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Plot failed: %s", exc)

    if a.backtest_origins:
        bt = E.backtest_margins(pan, origins=a.backtest_origins)
        (out / "backtest").mkdir(exist_ok=True)
        bt.to_csv(out / "backtest" / "backtest_margins.csv", index=False)
        bt.groupby(["method", "horizon"])[["wape_x_cp", "wape_va_cp", "share_ape_gt_20pct"]].mean() \
          .to_csv(out / "backtest" / "backtest_by_horizon.csv")
    with open(out / "run_notes.json", "w") as fh:
        json.dump({"methods": a.methods, "notes": notes, "split": bool(plan),
                   "dc_driver": a.dc_driver}, fh, indent=2)
    LOG.info("Projection written to %s", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-panel")
    b.add_argument("--out", type=Path, default=Path("data/projection/panel"))
    b.add_argument("--cp-dir", type=Path, default=None, help="OECD ICIO csv files (default: econ RData via model args)")
    b.add_argument("--pyp-dir", type=Path, default=None, help="OECD PYP ICIO csv files")
    b.add_argument("--pattern", default="*.csv")
    f = sub.add_parser("fetch-external")
    f.add_argument("--out", type=Path, default=Path("data/projection/external.csv"))
    f.add_argument("--ieo-route", default="ieo/2023")
    f.add_argument("--aeo-route", default="aeo/2026")
    r = sub.add_parser("run")
    r.add_argument("--panel", type=Path, default=Path("data/projection/panel"))
    r.add_argument("--external", type=Path, default=Path("data/projection/external.csv"))
    r.add_argument("--out", type=Path, default=Path("output/projection"))
    r.add_argument("--methods", nargs="+", default=["oecd", "arima", "external"],
                   choices=["oecd", "arima", "external"])
    r.add_argument("--base-csv", type=Path, default=None, help="2022 OECD wide csv instead of the RData")
    r.add_argument("--dc-driver", choices=["scenario", "parent"], default="scenario")
    r.add_argument("--backtest-origins", type=int, nargs="*", default=[2010, 2014, 2017])
    r.add_argument("--no-tables", dest="write_tables", action="store_false")
    r.add_argument("--rdata", dest="write_rdata", action="store_true", help="also write A/X/FD RData per year")
    a, rest = ap.parse_known_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"build-panel": cmd_build_panel, "fetch-external": cmd_fetch_external, "run": cmd_run}[a.cmd](a, rest)


if __name__ == "__main__":
    main()
