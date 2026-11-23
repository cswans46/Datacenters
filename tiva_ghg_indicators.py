#!/usr/bin/env python3
"""
TiVA and GHG-footprint indicators on the (Canada-split) ICIO.

Follows the OECD TiVA (2025 edition, principal indicators) and ICIO GHG
footprint (GHGFP 2025) definitions, computed directly from the model's
A, X and FD so that they also cover the split Canadian data-centre
sectors, and using the repo's hybrid emissions extension for emissions
intensities.

Notation
--------
    B = (I - A)^-1                Leontief inverse
    v = value-added coefficients   (1 - column sums of A unless VA given;
                                    this includes net taxes on intermediate
                                    products, as in the ICIO "TLS" row)
    f = direct GHG intensity       (t CO2e per USD mn of output)
    e_c = gross exports of country c (intermediate + final, to all others)
    y_d = final demand of country d (all origins)

Country indicators (TiVA names in brackets)
    EXGR        gross exports
    EXGR_DVA    domestic VA in gross exports            sum_{i in c} v_i (B e_c)_i
    EXGR_FVA    foreign VA in gross exports             sum_{i not in c} v_i (B e_c)_i
    EXGR_DVAFX  c's VA in other countries' exports      sum_{d != c} sum_{i in c} v_i (B e_d)_i
                (forward participation; TiVA EXGR_INDIR-type measure)
    FFD_DVA     c's VA in foreign final demand          sum_{d != c} sum_{i in c} v_i (B y_d)_i
    The GHG analogues replace v with f:
    EXGR_DGHG, EXGR_FGHG, EXGR_DGHGFX, PROD_GHG (f'x), DEMAND_GHG
    (footprint, sum_i f_i (B y_c)_i), GHG embodied in final-demand imports
    and exports, and the production-minus-footprint balance.

Canadian sector indicators (all CAN sectors, including CAN_J62_63_CLD,
_DPH and _OTH when the split is active)
    backward: domestic and foreign VA and GHG embodied per USD of the
              sector's exports, and in total ("foreign products used in
              Canadian exports")
    forward:  the sector's VA and GHG embodied in other countries' exports
              and in foreign final demand ("Canadian products used in
              foreign exports"), plus the Ghosh forward GHG multiplier.

Gross-exports measures count intermediate goods each time they cross a
border, as in the OECD TiVA EXGR_* family; they are not the Koopman-
Wang-Wei decomposition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

from canada_dsut_split import country_of, sector_of


LOG = logging.getLogger("datacentre_icio.indicators")


def _indicator(labels: list[str], groups: list[str]) -> np.ndarray:
    pos = {g: i for i, g in enumerate(groups)}
    m = np.zeros((len(labels), len(groups)))
    for r, lab in enumerate(labels):
        m[r, pos[lab]] = 1.0
    return m


def compute_indicators(
    A: pd.DataFrame,
    X: pd.Series,
    FD: pd.DataFrame,
    intensity: pd.Series,
    focus_country: str = "CAN",
    focus_keys: Optional[list[str]] = None,
    VA: Optional[pd.Series] = None,
) -> dict[str, pd.DataFrame]:
    keys = A.index.astype(str).tolist()
    n = len(keys)
    if A.columns.astype(str).tolist() != keys:
        raise ValueError("A must have identical row and column ordering.")

    a = A.to_numpy(dtype=float)
    x = X.reindex(keys).to_numpy(dtype=float)
    fd = FD.reindex(keys).to_numpy(dtype=float)
    f = intensity.reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(f).all():
        raise ValueError("Emissions intensities are missing for some ICIO industries.")

    if VA is not None:
        va = VA.reindex(keys).to_numpy(dtype=float)
        v = np.divide(va, x, out=np.zeros_like(va), where=x > 0)
    else:
        v = 1.0 - a.sum(axis=0)
        v[x <= 0] = 0.0
    if (v < -1e-9).any():
        LOG.warning("%s industries have negative value-added coefficients.", int((v < -1e-9).sum()))

    known = {country_of(k) for k in keys}
    row_c = [country_of(k, known) for k in keys]
    fd_c = [country_of(c, known) for c in FD.columns.astype(str)]
    unknown_fd = sorted(set(fd_c) - known)
    if unknown_fd:
        raise ValueError(f"Could not assign final-demand columns to ICIO countries: {unknown_fd[:10]}")

    countries = sorted(known)
    C = len(countries)
    cidx = {c: i for i, c in enumerate(countries)}
    row_ci = np.array([cidx[c] for c in row_c])
    S_row = _indicator(row_c, countries)     # n x C
    S_fd = _indicator(fd_c, countries)       # nfd x C

    Z = a * x[None, :]
    Z_by_user = Z @ S_row                    # n x C : sales to each user country (intermediate)
    FD_by_user = fd @ S_fd                   # n x C : sales to each final-demand country
    own = np.arange(n), row_ci
    exports = (Z_by_user.sum(axis=1) - Z_by_user[own]) + (FD_by_user.sum(axis=1) - FD_by_user[own])

    LOG.info("Factorizing Leontief system for indicators (%s industries).", n)
    lu = lu_factor(np.eye(n) - a)

    E = exports[:, None] * S_row             # n x C : exports of each country
    BE = lu_solve(lu, E)
    BY = lu_solve(lu, FD_by_user)            # n x C : output induced by each country's FD

    va_exp = S_row.T @ (v[:, None] * BE)     # origin country x exporter country
    ghg_exp = S_row.T @ (f[:, None] * BE)
    va_fd = S_row.T @ (v[:, None] * BY)      # origin country x demanding country
    ghg_fd = S_row.T @ (f[:, None] * BY)

    diag = np.eye(C, dtype=bool)
    exgr = E.sum(axis=0)

    tiva = pd.DataFrame({
        "country": countries,
        "EXGR_usd_mn": exgr,
        "EXGR_DVA_usd_mn": va_exp[diag],
        "EXGR_FVA_usd_mn": va_exp.sum(axis=0) - va_exp[diag],
        "EXGR_DVAFX_usd_mn": va_exp.sum(axis=1) - va_exp[diag],
        "FFD_DVA_usd_mn": va_fd.sum(axis=1) - va_fd[diag],
        "DDC_DVA_usd_mn": va_fd[diag],
        "FD_total_usd_mn": FD_by_user.sum(axis=0),
    })
    tiva["EXGR_DVA_share"] = tiva["EXGR_DVA_usd_mn"] / tiva["EXGR_usd_mn"].replace(0, np.nan)
    tiva["EXGR_FVA_share"] = tiva["EXGR_FVA_usd_mn"] / tiva["EXGR_usd_mn"].replace(0, np.nan)
    tiva["va_accounting_gap_usd_mn"] = tiva["EXGR_DVA_usd_mn"] + tiva["EXGR_FVA_usd_mn"] - tiva["EXGR_usd_mn"]

    prod_ghg = S_row.T @ (f * x)
    ghg = pd.DataFrame({
        "country": countries,
        "PROD_GHG_t": prod_ghg,
        "DEMAND_GHG_t": ghg_fd.sum(axis=0),
        "GHG_in_FD_imports_t": ghg_fd.sum(axis=0) - ghg_fd[diag],
        "GHG_in_FD_exports_t": ghg_fd.sum(axis=1) - ghg_fd[diag],
        "EXGR_DGHG_t": ghg_exp[diag],
        "EXGR_FGHG_t": ghg_exp.sum(axis=0) - ghg_exp[diag],
        "EXGR_DGHGFX_t": ghg_exp.sum(axis=1) - ghg_exp[diag],
        "EXGR_usd_mn": exgr,
    })
    ghg["BALANCE_PROD_minus_DEMAND_t"] = ghg["PROD_GHG_t"] - ghg["DEMAND_GHG_t"]
    ghg["EXGR_GHG_intensity_t_per_usd_mn"] = (ghg["EXGR_DGHG_t"] + ghg["EXGR_FGHG_t"]) / ghg["EXGR_usd_mn"].replace(0, np.nan)

    bilateral = pd.DataFrame({
        "origin_country": np.repeat(countries, C),
        "partner_country": np.tile(countries, C),
        "va_in_partner_gross_exports_usd_mn": va_exp.ravel(),
        "ghg_in_partner_gross_exports_t": ghg_exp.ravel(),
        "va_in_partner_final_demand_usd_mn": va_fd.ravel(),
        "ghg_in_partner_final_demand_t": ghg_fd.ravel(),
    })

    # ---------------- Canadian sectors ---------------------------------
    if focus_country not in cidx:
        raise ValueError(f"Focus country {focus_country} not in ICIO.")
    home = cidx[focus_country]
    home_idx = np.where(row_ci == home)[0]
    home_keys = [keys[i] for i in home_idx]
    home_row = row_ci == home

    # Backward: B[:, j] for every home sector j.
    unit = np.zeros((n, len(home_idx)))
    unit[home_idx, np.arange(len(home_idx))] = 1.0
    Bh = lu_solve(lu, unit)                                 # n x H
    ghg_mult_dom = (f[home_row, None] * Bh[home_row]).sum(axis=0)
    ghg_mult_for = (f[~home_row, None] * Bh[~home_row]).sum(axis=0)
    va_mult_dom = (v[home_row, None] * Bh[home_row]).sum(axis=0)
    va_mult_for = (v[~home_row, None] * Bh[~home_row]).sum(axis=0)
    exp_h = exports[home_idx]

    # Forward: home-sector VA/GHG embodied in other countries' exports and FD.
    foreign_cols = np.arange(C) != home
    va_fx = (v[home_idx, None] * BE[home_idx][:, foreign_cols]).sum(axis=1)
    ghg_fx = (f[home_idx, None] * BE[home_idx][:, foreign_cols]).sum(axis=1)
    va_ffd = (v[home_idx, None] * BY[home_idx][:, foreign_cols]).sum(axis=1)
    ghg_ffd = (f[home_idx, None] * BY[home_idx][:, foreign_cols]).sum(axis=1)

    # Ghosh forward multiplier: (I - C)^-1 f, C = diag(1/x) Z.
    Cmat = np.divide(Z, x[:, None], out=np.zeros_like(Z), where=x[:, None] > 0)
    ghosh_f = np.linalg.solve(np.eye(n) - Cmat, f)

    sectors = pd.DataFrame({
        "io_key": home_keys,
        "sector_code": [sector_of(k) for k in home_keys],
        "output_usd_mn": x[home_idx],
        "gross_exports_usd_mn": exp_h,
        "direct_ghg_intensity_t_per_usd_mn": f[home_idx],
        # backward
        "backward_va_multiplier_domestic": va_mult_dom,
        "backward_va_multiplier_foreign": va_mult_for,
        "backward_ghg_multiplier_domestic_t_per_usd_mn": ghg_mult_dom,
        "backward_ghg_multiplier_foreign_t_per_usd_mn": ghg_mult_for,
        "exports_domestic_va_usd_mn": va_mult_dom * exp_h,
        "exports_foreign_va_usd_mn": va_mult_for * exp_h,
        "exports_domestic_ghg_t": ghg_mult_dom * exp_h,
        "exports_foreign_ghg_t": ghg_mult_for * exp_h,
        # forward
        "va_in_foreign_gross_exports_usd_mn": va_fx,
        "ghg_in_foreign_gross_exports_t": ghg_fx,
        "va_in_foreign_final_demand_usd_mn": va_ffd,
        "ghg_in_foreign_final_demand_t": ghg_ffd,
        "forward_ghosh_ghg_multiplier_t_per_usd_mn": ghosh_f[home_idx],
    })

    # Partner detail for the focus keys.
    if focus_keys is None:
        focus_keys = [k for k in home_keys if sector_of(k).startswith("J62")]
    pos_home = {k: p for p, k in enumerate(home_keys)}
    detail = []
    for key in focus_keys:
        if key not in pos_home:
            raise ValueError(f"Focus key {key} is not a {focus_country} industry.")
        i = keys.index(key)
        p = pos_home[key]
        back_va = S_row.T @ (v * Bh[:, p] * exp_h[p])
        back_ghg = S_row.T @ (f * Bh[:, p] * exp_h[p])
        for ci, c in enumerate(countries):
            detail.append({
                "focus_io_key": key,
                "partner_country": c,
                # forward: focus sector's content in partner's exports / final demand
                "forward_va_in_partner_gross_exports_usd_mn": v[i] * BE[i, ci],
                "forward_ghg_in_partner_gross_exports_t": f[i] * BE[i, ci],
                "forward_va_in_partner_final_demand_usd_mn": v[i] * BY[i, ci],
                "forward_ghg_in_partner_final_demand_t": f[i] * BY[i, ci],
                # backward: partner-origin content in the focus sector's exports
                "backward_partner_va_in_focus_exports_usd_mn": back_va[ci],
                "backward_partner_ghg_in_focus_exports_t": back_ghg[ci],
            })
    focus = pd.DataFrame(detail)

    checks = pd.DataFrame([{
        "max_abs_va_accounting_gap_usd_mn": float(tiva["va_accounting_gap_usd_mn"].abs().max()),
        "world_prod_ghg_t": float(prod_ghg.sum()),
        "world_demand_ghg_t": float(ghg_fd.sum()),
        "world_va_in_fd_usd_mn": float(va_fd.sum()),
        "world_fd_usd_mn": float(FD_by_user.sum()),
    }])

    return {
        "tiva_country_indicators": tiva,
        "ghg_footprint_country_indicators": ghg,
        "bilateral_origin_partner": bilateral,
        f"{focus_country.lower()}_sector_forward_backward": sectors,
        f"{focus_country.lower()}_focus_sectors_by_partner": focus,
        "indicator_checks": checks,
    }


def write_indicators(results: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    LOG.info("Wrote TiVA and GHG footprint indicators to %s", out_dir)
