from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .attention_control_representation import (
    fit_attention_directions as _attention_fit_directions,
    load_token_table as _load_token_table,
    pool_attention_tokens as _attention_family_pool,
)
from .short_contact_benchmark import LABEL_TO_INT, _select_c_inner

FEATURE_PREFIX = "feat_"


def load_pooled_table(path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Load a pooled feature table keyed by (uid, window_name)."""
    frame = pd.read_csv(path)
    feature_columns = [
        column for column in frame if column.startswith(FEATURE_PREFIX)
    ]
    result: dict[tuple[str, str], np.ndarray] = {}
    for (uid, window_name), group in frame.groupby(["uid", "window_name"]):
        ordered = group.sort_values("uid")
        values = ordered[feature_columns].to_numpy(dtype=np.float64)
        result[(str(uid), str(window_name))] = (
            values[0] if values.shape[0] == 1 else values
        )
    return result


def load_source_table(
    path: Path, is_attention: bool
) -> dict[tuple[str, str], np.ndarray]:
    if is_attention:
        return _load_token_table(path)
    return load_pooled_table(path)


def verify_fold_consistency(
    fold_a: pd.DataFrame,
    fold_b: pd.DataFrame,
    name_a: str = "fold_a",
    name_b: str = "fold_b",
) -> None:
    """Verify two fold assignments agree on every shared uid.

    Raises ValueError naming the first mismatched uids. Uids present in
    only one table are allowed (per-source eligibility differs); the
    assignments themselves must agree.
    """
    by_uid_a = fold_a.set_index("uid")["outer_fold"]
    by_uid_b = fold_b.set_index("uid")["outer_fold"]
    shared = by_uid_a.index.intersection(by_uid_b.index)
    mismatched = shared[
        by_uid_a[shared].to_numpy() != by_uid_b[shared].to_numpy()
    ]
    if len(mismatched) > 0:
        sample = ", ".join(str(value) for value in mismatched[:5])
        raise ValueError(
            f"Fold assignments disagree between {name_a} and {name_b} "
            f"for {len(mismatched)} shared uids: {sample}"
        )


def _source_event_uids(
    table: dict[tuple[str, str], np.ndarray],
    window_name: str,
) -> set[str]:
    return {uid for (uid, name) in table if name == window_name}


def _source_matrix(
    table: dict[tuple[str, str], np.ndarray],
    uids: list[str],
    window_name: str,
    is_attention: bool,
    labels: np.ndarray,
    train: np.ndarray,
    attention_k: int,
) -> np.ndarray:
    if not is_attention:
        return np.stack(
            [table[(uid, window_name)] for uid in uids], axis=0
        )
    directions = _attention_fit_directions(
        [table[(uids[i], window_name)] for i in train],
        labels[train],
        "attention",
        attention_k,
    )
    return np.stack(
        [
            _attention_family_pool(
                table[(uid, window_name)],
                directions,
                "attention",
                attention_k,
            )
            for uid in uids
        ]
    )


def _evaluate_combination(
    matrices_by_fold: dict[int, np.ndarray],
    pre_matrices_by_fold: dict[int, np.ndarray] | None,
    labels: np.ndarray,
    groups: np.ndarray,
    fold_array: np.ndarray,
    pre_labels: np.ndarray,
    pre_fold_array: np.ndarray | None,
    name: str,
    c_grid: tuple[float, ...],
    seed: int,
    inner_splits: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    folds = sorted(matrices_by_fold)
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    predictions = np.full(len(labels), -1, dtype=int)
    selection_rows: list[dict[str, object]] = []
    selected_c: dict[int, float] = {}
    for outer_fold in folds:
        test = np.flatnonzero(fold_array == outer_fold)
        train = np.flatnonzero(fold_array != outer_fold)
        matrix = matrices_by_fold[outer_fold]
        c_value, records = _select_c_inner(
            matrix,
            labels,
            groups,
            train,
            c_grid,
            inner_splits,
            seed + int(outer_fold),
        )
        selected_c[int(outer_fold)] = c_value
        scaler = StandardScaler()
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed + int(outer_fold),
        )
        classifier.fit(scaler.fit_transform(matrix[train]), labels[train])
        test_scores = classifier.predict_proba(
            scaler.transform(matrix[test])
        )[:, 1]
        scores[test] = test_scores
        predictions[test] = (test_scores >= 0.5).astype(int)
        selection_rows.append(
            {
                "name": name,
                "condition": "event_selected_event",
                "outer_fold": int(outer_fold),
                "selected_C": c_value,
                "inner_scores_json": json.dumps(records, sort_keys=True),
            }
        )
    if not np.isfinite(scores).all():
        raise AssertionError(f"Non-finite scores for {name}")
    event_ba = float(balanced_accuracy_score(labels, predictions))
    event_auc = float(roc_auc_score(labels, scores))
    row: dict[str, object] = {
        "name": name,
        "condition": "event_selected_event",
        "balanced_accuracy": event_ba,
        "roc_auc": event_auc,
        "eligible_samples": len(labels),
    }
    if pre_matrices_by_fold is not None:
        if pre_fold_array is None:
            raise AssertionError("pre matrices need the pre fold array")
        pre_scores = np.full(len(pre_labels), np.nan, dtype=np.float64)
        for outer_fold in folds:
            test = np.flatnonzero(pre_fold_array == outer_fold)
            train = np.flatnonzero(fold_array != outer_fold)
            matrix = matrices_by_fold[outer_fold]
            scaler = StandardScaler()
            classifier = LogisticRegression(
                C=selected_c[int(outer_fold)],
                class_weight="balanced",
                solver="liblinear",
                max_iter=5_000,
                random_state=seed + int(outer_fold),
            )
            classifier.fit(
                scaler.fit_transform(matrix[train]), labels[train]
            )
            pre_scores[test] = classifier.predict_proba(
                scaler.transform(pre_matrices_by_fold[outer_fold][test])
            )[:, 1]
        pre_ba = float(
            balanced_accuracy_score(
                pre_labels, (pre_scores >= 0.5).astype(int)
            )
        )
        row["pre_transfer_balanced_accuracy"] = pre_ba
        row["contact_specific_increment"] = event_ba - pre_ba
    return row, selection_rows


def evaluate_fusion(
    source_names: tuple[str, ...],
    source_tables: tuple[dict[tuple[str, str], np.ndarray], ...],
    attention_flags: tuple[bool, ...],
    folds: pd.DataFrame,
    c_grid: tuple[float, ...],
    seed: int,
    inner_splits: int = 3,
    event_name: str = "event_200ms",
    pre_name: str = "pre_200ms",
) -> dict[str, object]:
    """Evaluate single sources and their concatenation on one sample set.

    The sample set is the intersection of every source's event-window
    uids; the pre-transfer probe uses the same intersection restricted to
    uids with a strict-pre window. C is selected inside training folds
    only. Returns rows for every combination (each source alone plus the
    full concatenation) at event/pre/increment, plus a summary.
    """
    if len(source_names) < 2:
        raise ValueError("evaluate_fusion needs at least two sources")
    if not (len(source_names) == len(source_tables) == len(attention_flags)):
        raise ValueError("source names/tables/flags must be aligned")

    event_uids = sorted(
        set.intersection(
            *[
                _source_event_uids(table, event_name)
                for table in source_tables
            ]
        )
    )
    pre_uids = sorted(
        set.intersection(
            *[
                _source_event_uids(table, pre_name)
                for table in source_tables
            ]
        ).intersection(event_uids)
    )
    if len(event_uids) < 2:
        raise ValueError("Too few shared event uids for fusion evaluation")

    aligned = folds[folds["uid"].isin(event_uids)].sort_values(
        "uid"
    ).reset_index(drop=True)
    aligned_uids = [str(uid) for uid in aligned["uid"]]
    labels = aligned["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = aligned["lineage_group_id"].to_numpy(dtype=object)
    fold_array = aligned["outer_fold"].to_numpy(dtype=int)
    pre_aligned = folds[folds["uid"].isin(pre_uids)].sort_values(
        "uid"
    ).reset_index(drop=True)
    pre_uids_sorted = [str(uid) for uid in pre_aligned["uid"]]
    pre_labels = pre_aligned["label"].map(LABEL_TO_INT).to_numpy(dtype=int)

    fold_set = sorted(set(fold_array))
    folds_for = {
        fold: np.flatnonzero(fold_array == fold) for fold in fold_set
    }
    trains = {
        fold: np.flatnonzero(fold_array != fold) for fold in fold_set
    }
    pre_fold_array = pre_aligned["outer_fold"].to_numpy(dtype=int)
    pre_folds_for = {
        fold: np.flatnonzero(pre_fold_array == fold)
        for fold in fold_set
    }
    pre_trains = {
        fold: np.flatnonzero(pre_fold_array != fold)
        for fold in fold_set
    }

    def matrices_for(
        table: dict[tuple[str, str], np.ndarray],
        is_attention: bool,
        uid_list: list[str],
        window_name: str,
        label_array: np.ndarray,
        train_lookup: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        if not is_attention:
            matrix = _source_matrix(
                table, uid_list, window_name, False, label_array,
                train_lookup[fold_set[0]], 3,
            )
            return {fold: matrix for fold in fold_set}
        return {
            fold: _source_matrix(
                table,
                uid_list,
                window_name,
                True,
                label_array,
                train_lookup[fold],
                3,
            )
            for fold in fold_set
        }

    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for index, (name, table, is_attention) in enumerate(
        zip(source_names, source_tables, attention_flags, strict=True)
    ):
        event_matrices = matrices_for(
            table, is_attention, aligned_uids, event_name, labels, trains
        )
        pre_matrices = (
            matrices_for(
                table,
                is_attention,
                pre_uids_sorted,
                pre_name,
                pre_labels,
                pre_trains,
            )
            if pre_uids
            else None
        )
        row, selections = _evaluate_combination(
            event_matrices,
            pre_matrices,
            labels,
            groups,
            fold_array,
            pre_labels,
            pre_fold_array if pre_uids else None,
            name,
            c_grid,
            seed,
            inner_splits,
        )
        rows.append(row)
        selection_rows.extend(selections)

    combined_name = "+".join(source_names)
    event_matrices = {
        fold: np.concatenate(
            [
                matrices_for(
                    table, is_attention, aligned_uids, event_name, labels,
                    trains,
                )[fold]
                for table, is_attention in zip(
                    source_tables, attention_flags, strict=True
                )
            ],
            axis=1,
        )
        for fold in fold_set
    }
    pre_matrices = (
        {
            fold: np.concatenate(
                [
                    matrices_for(
                        table, is_attention, pre_uids_sorted, pre_name,
                        pre_labels, pre_trains,
                    )[fold]
                    for table, is_attention in zip(
                        source_tables, attention_flags, strict=True
                    )
                ],
                axis=1,
            )
            for fold in fold_set
        }
        if pre_uids
        else None
    )
    row, selections = _evaluate_combination(
        event_matrices,
        pre_matrices,
        labels,
        groups,
        fold_array,
        pre_labels,
        pre_fold_array if pre_uids else None,
        combined_name,
        c_grid,
        seed,
        inner_splits,
    )
    rows.append(row)
    selection_rows.extend(selections)

    table = pd.DataFrame(rows)
    summary = {
        "sample_set": "intersection_of_source_event_uids",
        "event_eligible_samples": len(aligned_uids),
        "pre_eligible_samples": len(pre_uids_sorted),
        "sources": list(source_names),
        "c_grid": list(c_grid),
        "seed": seed,
    }
    return {"table": table, "selections": selection_rows, "summary": summary}
