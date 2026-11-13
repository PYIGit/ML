"""짝(값이 똑같은 다른 부품)의 조건 계산 (4-3 문서).

묶음 번호는 "값이 전부 같은 부품끼리 같은 번호"로 이미 붙어 있다고 본다(features.confirm_row_groups).
여기서는 같은 묶음 안에서 조건을 더 건다: Id가 바로 이웃인가, 시각이 전부 같은가.
"""
import numpy as np

from ml.data.features import confirm_row_groups


def has_adjacent_id(group: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """행마다: 같은 묶음 안에 Id가 1 차이 나는 다른 행이 있는가. 묶음 번호가 음수인 행은 False."""
    group = np.asarray(group)
    ids = np.asarray(ids)
    out = np.zeros(len(group), dtype=bool)
    if len(group) < 2:
        return out
    order = np.lexsort((ids, group))
    g, i = group[order], ids[order]
    adj = (g[1:] == g[:-1]) & (i[1:] - i[:-1] == 1) & (g[1:] >= 0)
    out[order[1:][adj]] = True
    out[order[:-1][adj]] = True
    return out


def has_same_times(group: np.ndarray, times: np.ndarray) -> np.ndarray:
    """행마다: 같은 묶음 안에 시각이 전부 같은 다른 행이 있는가.

    times: 행 x 시각 열(빈 칸은 NaN). 빈 칸끼리는 같다고 본다. 시각이 하나도 없는 행은 False.
    묶음 번호가 음수인 행은 False.
    """
    group = np.asarray(group)
    out = np.zeros(len(group), dtype=bool)
    ok = (group >= 0) & ~np.isnan(times).all(axis=1)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return out
    sub = confirm_row_groups(group[idx], times[idx])
    out[idx[sub >= 0]] = True
    return out


def has_mate_within(group: np.ndarray, member: np.ndarray) -> np.ndarray:
    """행마다: member인 행 가운데 같은 묶음의 다른 member 행이 있는가(예: train 부품끼리의 짝)."""
    group = np.asarray(group)
    member = np.asarray(member, dtype=bool) & (group >= 0)
    out = np.zeros(len(group), dtype=bool)
    if not member.any():
        return out
    n = np.bincount(group[member], minlength=int(group.max()) + 1)
    out[member] = n[group[member]] >= 2
    return out
