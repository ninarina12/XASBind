"""Tests for grouped_stratified_split.

These guard against the two bugs this dataset already hit once: index sets that
overlap, and group (material) leakage across splits.
"""
import numpy as np
import pandas as pd
import pytest

from xasbind.data.splits import grouped_stratified_split


def _toy_df(n_materials: int = 80, per_material: int = 4, seed: int = 0) -> pd.DataFrame:
    elements = ["O", "Li", "Fe", "P", "S"]
    rows = []
    for m in range(n_materials):
        elem = elements[m % len(elements)]  # balanced -> stratifiable
        for _ in range(per_material):
            rows.append({"material_id": f"mp-{m}", "absorbing_element": elem})
    return pd.DataFrame(rows)


def test_disjoint_and_full_coverage():
    df = _toy_df()
    train, valid, test, _ = grouped_stratified_split(
        df, valid_size=0.2, test_size=0.1, random_state=0)
    assert set(train).isdisjoint(valid)
    assert set(train).isdisjoint(test)
    assert set(valid).isdisjoint(test)
    assert len(train) + len(valid) + len(test) == len(df)


def test_no_group_leakage():
    df = _toy_df().reset_index(drop=True)
    train, valid, test, _ = grouped_stratified_split(
        df, valid_size=0.2, test_size=0.1, random_state=0)

    def mats(idx):
        return set(df.loc[idx, "material_id"])

    assert mats(train).isdisjoint(mats(valid))
    assert mats(train).isdisjoint(mats(test))
    assert mats(valid).isdisjoint(mats(test))


def test_explicit_n_splits():
    df = _toy_df()
    train, valid, test, n_splits = grouped_stratified_split(
        df, valid_size=0.2, test_size=0.1, n_splits=20, random_state=0)
    assert n_splits == 20
    assert len(train) + len(valid) + len(test) == len(df)


def test_invalid_fractions_raise():
    df = _toy_df()
    with pytest.raises(ValueError):
        grouped_stratified_split(df, valid_size=0.6, test_size=0.6)
