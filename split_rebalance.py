#!/usr/bin/env python3
"""
RAS re-balancing of split ICIO sectors and table-wide balance checks.

When an ICIO industry (the parent) is split into subsectors, its column of
intermediate inputs is shared out across the subsector columns. The split is
exact in aggregate (the subsector columns add back to the parent column), but
how much each subsector spends on inputs, and therefore its value added, is
then only a residual. ``ras_split_columns`` fixes this with a biproportional
(RAS) balance of the n x K block of subsector columns:

    row targets      sum_k Z[i, k] = Z[i, parent]            (every input row i)
    column targets   sum_i Z[i, k] = x_k - VA_k*             (every subsector k)

with value-added targets taken from the source table (StatCan SUT for Canada,
the 2023-24 India IOT for India) and scaled so they add up to the parent's
value added:

    VA_k* = VA_parent * x_k s_k / sum_m x_m s_m ,

where s_k is the subsector's value-added share of output in the source
table. Every other cell of the ICIO is untouched, so the whole table stays
balanced: rows still sum to output (x = Z1 + f) and columns still sum to
output (x' = 1'Z + VA'), with the parent's value added preserved.

The signed RAS routine (icio_projection.ras.ras_signed, from
cswans46/baltic) holds negative seed cells fixed.

``balance_report`` checks the full table after all splits and overrides:
the row identity, and industries with negative value added.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from icio_projection.ras import ras_signed

LOG = logging.getLogger("datacentre_icio.split_rebalance")


def scaled_va_targets(parent_va: float, x_sub: np.ndarray, va_share: np.ndarray) -> np.ndarray:
    """Subsector value-added targets that add up to the parent's value added."""
    x_sub = np.asarray(x_sub, dtype=float)
    va_share = np.clip(np.asarray(va_share, dtype=float), 0.0, None)
    weight = x_sub * va_share
    if weight.sum() <= 0:
        return parent_va * x_sub / x_sub.sum()
    return parent_va * weight / weight.sum()


def ras_split_columns(
    seed_columns: np.ndarray,
    parent_column: np.ndarray,
    x_sub: np.ndarray,
    va_targets: np.ndarray,
    label: str = "split columns",
) -> tuple[np.ndarray, dict]:
    """
    Biproportionally balance the n x K subsector input block.

    seed_columns   n x K first-guess input values of the subsectors
    parent_column  n parent input values (row targets)
    x_sub          K subsector outputs
    va_targets     K subsector value-added targets (sum = parent VA)
    """
    seed = np.asarray(seed_columns, dtype=float)
    parent = np.asarray(parent_column, dtype=float)
    x_sub = np.asarray(x_sub, dtype=float)
    va_targets = np.asarray(va_targets, dtype=float)

    col_targets = x_sub - va_targets
    if (col_targets < -1e-9).any():
        raise ValueError(f"{label}: value-added targets exceed subsector output.")
    if not np.isclose(col_targets.sum(), parent.sum(), rtol=1e-6, atol=1e-6 * max(1.0, abs(parent.sum()))):
        raise ValueError(
            f"{label}: column targets ({col_targets.sum():.3f}) do not add up to the "
            f"parent's intermediate inputs ({parent.sum():.3f})."
        )

    # Strictly positive seed where the parent buys something, so every input
    # can be shared between the subsectors.
    seed = np.where(parent[:, None] > 0, np.maximum(seed, 0.0), np.minimum(seed, 0.0))
    tiny = 1e-12 * max(1.0, float(np.abs(parent).max()))
    seed = np.where((parent[:, None] > 0) & (seed <= 0), tiny, seed)

    balanced, info = ras_signed(seed, parent, col_targets, label=label)

    row_err = float(np.abs(balanced.sum(axis=1) - parent).max())
    col_err = float(np.abs(balanced.sum(axis=0) - col_targets).max())
    info.update({"max_row_error": row_err, "max_col_error": col_err})
    if row_err > 1e-6 * max(1.0, float(np.abs(parent).max())) or col_err > 1e-6 * max(1.0, float(x_sub.max())):
        LOG.warning("%s: RAS residual errors row %.3e, column %.3e", label, row_err, col_err)
    return balanced, info


def balance_report(
    A: pd.DataFrame,
    X: pd.Series,
    FD: pd.DataFrame,
    tolerance: float = 1e-3,
) -> tuple[dict, pd.DataFrame]:
    """
    Table-wide checks after splitting and re-balancing.

    Returns a summary dict and a frame of industries with negative value
    added (value-added coefficient 1 - column sum of A below zero).
    """
    keys = A.index.astype(str).tolist()
    a = A.loc[keys, keys].to_numpy(dtype=float)
    x = X.reindex(keys).to_numpy(dtype=float)
    fd = FD.reindex(keys).to_numpy(dtype=float).sum(axis=1)

    row_total = a @ x + fd
    rel = np.abs(row_total - x) / np.maximum(np.abs(x), 1.0)
    va_coeff = 1.0 - a.sum(axis=0)
    va_coeff[x <= 0] = 0.0

    negative = pd.DataFrame({
        "io_key": keys,
        "output_usd_mn": x,
        "value_added_coefficient": va_coeff,
        "value_added_usd_mn": va_coeff * x,
    })
    negative = negative.loc[negative["value_added_coefficient"] < -1e-9].reset_index(drop=True)

    summary = {
        "industries": len(keys),
        "max_row_identity_relative_error": float(rel.max()),
        "row_identity_ok": bool(rel.max() <= tolerance),
        "negative_value_added_industries": int(len(negative)),
    }
    return summary, negative


def log_balance(summary: dict, negative: pd.DataFrame, label: str = "ICIO") -> None:
    LOG.info(
        "%s balance after splits: %s industries, max row-identity error %.3g.",
        label, summary["industries"], summary["max_row_identity_relative_error"],
    )
    if not summary["row_identity_ok"]:
        LOG.warning("%s: rows do not sum to gross output within tolerance.", label)
    if len(negative):
        LOG.warning(
            "%s: %s industries with negative value added: %s",
            label, len(negative),
            ", ".join(f"{k} ({v:.3f})" for k, v in zip(negative["io_key"], negative["value_added_coefficient"])),
        )


def split_emissions_by_basis(
    emissions: pd.DataFrame,
    parent_key: str,
    sub_keys: list[str],
    X_split: pd.Series,
    A_split: Optional[pd.DataFrame] = None,
    electricity_keys: Optional[list[str]] = None,
    basis: str = "output",
    label: str = "split",
) -> pd.DataFrame:
    """
    Split the parent's direct emissions row across subsectors.

    output             equal emissions intensity (by output)
    electricity_input  in proportion to purchases from electricity_keys
    """
    parent = emissions.loc[emissions["io_key"].eq(parent_key)]
    if len(parent) != 1:
        raise ValueError(f"Emissions extension must contain exactly one row for {parent_key}.")
    row = parent.iloc[0]
    x = X_split.reindex(sub_keys).to_numpy(dtype=float)

    if basis == "output":
        shares = x / x.sum()
    elif basis == "electricity_input":
        if A_split is None or not electricity_keys:
            raise ValueError("electricity_input basis requires the split A and electricity keys.")
        purchases = A_split.loc[electricity_keys, sub_keys].to_numpy(float).sum(axis=0) * x
        if purchases.sum() <= 0:
            raise ValueError("No electricity purchases in the split subsectors; use basis='output'.")
        shares = purchases / purchases.sum()
    else:
        raise ValueError(f"Unknown emissions split basis: {basis}")

    new_rows = []
    for key, share in zip(sub_keys, shares):
        new = row.copy()
        new["io_key"] = key
        new["ghg_tonnes_co2e"] = float(row["ghg_tonnes_co2e"]) * float(share)
        new["source"] = f"{row['source']}; {label}_{basis}"
        new_rows.append(new)

    rest = emissions.loc[~emissions["io_key"].eq(parent_key)]
    return pd.concat([rest, pd.DataFrame(new_rows)], ignore_index=True)
