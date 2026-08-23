from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_CONDITIONS = (
    "event_selected_event",
    "event_selected_pre",
    "pre_selected_pre",
    "event_selected_removed",
    "removed_selected_removed",
)
PRE_ONLY_CONDITIONS = CONTROL_CONDITIONS[:3]


def load_token_table(path: Path) -> dict[tuple[str, str], np.ndarray]:
    frame = pd.read_csv(path)
    feature_columns = [
        column for column in frame if column.startswith("feat_")
    ]
    result: dict[tuple[str, str], np.ndarray] = {}
    for (uid, window_name), group in frame.groupby(["uid", "window_name"]):
        ordered = group.sort_values("token_index")
        result[(str(uid), str(window_name))] = ordered[
            feature_columns
        ].to_numpy(dtype=np.float64)
    return result


def fit_attention_directions(
    token_rows: list[np.ndarray],
    sample_labels: np.ndarray,
    pooling: str,
    k: int,
) -> list[np.ndarray]:
    """Fit attention-family directions on one outer training fold."""

    tokens_flat = np.concatenate(token_rows, axis=0)
    labels_flat = np.repeat(
        sample_labels, [rows.shape[0] for rows in token_rows]
    )
    if pooling == "attention_lda":
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
        )

        if len(np.unique(labels_flat)) < 2:
            raise ValueError(
                "attention_lda requires both classes in the training fold"
            )
        lda = LinearDiscriminantAnalysis(
            n_components=1, solver="lsqr", shrinkage="auto"
        )
        lda.fit(tokens_flat, labels_flat)
        direction = lda.coef_[0].astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            return [direction / norm]
        centered = tokens_flat - tokens_flat.mean(axis=0)
        _singular, _values, vh = np.linalg.svd(
            centered, full_matrices=False
        )
        return [vh[0].astype(np.float64)]
    centered = tokens_flat - tokens_flat.mean(axis=0)
    _singular, _values, vh = np.linalg.svd(centered, full_matrices=False)
    count = k if pooling == "attention_multi" else 1
    return [vh[index].astype(np.float64) for index in range(count)]


def pool_attention_tokens(
    tokens: np.ndarray,
    directions: list[np.ndarray],
    pooling: str,
    k: int,
) -> np.ndarray:
    if pooling == "attention_neighbourhood":
        if tokens.shape[0] < 2:
            raise ValueError(
                "attention_neighbourhood requires at least two tokens"
            )
        logits = tokens @ directions[0]
        peak = int(np.argmax(logits))
        indices = [
            max(0, peak - 1),
            peak,
            min(tokens.shape[0] - 1, peak + 1),
        ]
        return np.concatenate([tokens[index] for index in indices])
    if pooling == "attention_multi":
        parts: list[np.ndarray] = []
        for direction in directions[:k]:
            logits = tokens @ direction
            shifted = logits - logits.max()
            weights = np.exp(shifted)
            weights = weights / weights.sum()
            parts.append(weights @ tokens)
        return np.concatenate(parts)
    logits = tokens @ directions[0]
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    weights = weights / weights.sum()
    return weights @ tokens


@dataclass(frozen=True)
class AttentionControlWindowRoles:
    window_names: tuple[str, ...]
    conditions: tuple[str, ...]
    fit_window_items: tuple[tuple[str, str], ...]
    apply_window_items: tuple[tuple[str, str], ...]

    @property
    def fit_window_by_condition(self) -> dict[str, str]:
        return dict(self.fit_window_items)

    @property
    def apply_window_by_condition(self) -> dict[str, str]:
        return dict(self.apply_window_items)


def attention_control_window_roles(
    event_window: str,
    pre_window: str,
    removed_window: str | None,
) -> AttentionControlWindowRoles:
    window_names = [event_window, pre_window]
    fit_windows = {
        "event_selected_event": event_window,
        "event_selected_pre": event_window,
        "pre_selected_pre": pre_window,
    }
    apply_windows = {
        "event_selected_event": event_window,
        "event_selected_pre": pre_window,
        "pre_selected_pre": pre_window,
    }
    conditions = PRE_ONLY_CONDITIONS
    if removed_window is not None:
        window_names.append(removed_window)
        fit_windows.update(
            {
                "event_selected_removed": event_window,
                "removed_selected_removed": removed_window,
            }
        )
        apply_windows.update(
            {
                "event_selected_removed": removed_window,
                "removed_selected_removed": removed_window,
            }
        )
        conditions = CONTROL_CONDITIONS
    return AttentionControlWindowRoles(
        window_names=tuple(window_names),
        conditions=conditions,
        fit_window_items=tuple(fit_windows.items()),
        apply_window_items=tuple(apply_windows.items()),
    )


@dataclass(frozen=True)
class AttentionControlRepresentation:
    token_table: dict[tuple[str, str], np.ndarray]
    paired: pd.DataFrame
    paired_uids: tuple[str, ...]
    roles: AttentionControlWindowRoles

    @classmethod
    def from_token_table(
        cls,
        token_table: dict[tuple[str, str], np.ndarray],
        folds: pd.DataFrame,
        roles: AttentionControlWindowRoles,
    ) -> "AttentionControlRepresentation":
        available = set.intersection(
            *(
                {
                    uid
                    for uid, candidate_window in token_table
                    if candidate_window == window_name
                }
                for window_name in roles.window_names
            )
        )
        paired = folds[folds["uid"].isin(available)].sort_values(
            "uid"
        ).reset_index(drop=True)
        paired_uids = tuple(str(uid) for uid in paired["uid"])
        dimensions: set[int] = set()
        for uid in paired_uids:
            for window_name in roles.window_names:
                tokens = token_table[(uid, window_name)]
                if (
                    tokens.ndim != 2
                    or not len(tokens)
                    or not np.isfinite(tokens).all()
                ):
                    raise ValueError(
                        f"Non-finite or malformed tokens for "
                        f"{uid}/{window_name}"
                    )
                dimensions.add(int(tokens.shape[1]))
        if len(dimensions) > 1:
            raise ValueError("Token feature dimensions are inconsistent")
        return cls(
            token_table=token_table,
            paired=paired,
            paired_uids=paired_uids,
            roles=roles,
        )

    def fold_matrices(
        self,
        train: np.ndarray,
        labels: np.ndarray,
        pooling: str,
        attention_k: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        directions_by_fit_window = {
            fit_window: fit_attention_directions(
                [
                    self.token_table[(self.paired_uids[position], fit_window)]
                    for position in train
                ],
                labels[train],
                pooling,
                attention_k,
            )
            for fit_window in self.roles.window_names
        }

        def pool_with_fit_window(
            apply_window: str,
            fit_window: str,
        ) -> np.ndarray:
            directions = directions_by_fit_window[fit_window]
            matrix = np.stack(
                [
                    pool_attention_tokens(
                        self.token_table[(uid, apply_window)],
                        directions,
                        pooling,
                        attention_k,
                    )
                    for uid in self.paired_uids
                ]
            )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"Attention pooling produced non-finite "
                    f"{apply_window} features"
                )
            return matrix

        source_matrices = {
            window_name: pool_with_fit_window(window_name, window_name)
            for window_name in self.roles.window_names
        }
        fit_windows = self.roles.fit_window_by_condition
        apply_windows = self.roles.apply_window_by_condition
        condition_matrices = {
            condition: pool_with_fit_window(
                apply_windows[condition], fit_windows[condition]
            )
            for condition in self.roles.conditions
        }
        return source_matrices, condition_matrices
