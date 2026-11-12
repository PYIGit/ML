"""예측 시점과 train·test 비교에 쓰는 계산 (3-6 문서).

가정: 스테이션 번호 순서가 거치는 순서다(3-4에서 확인). 스테이션 열은 번호순으로 준다.
"""
import numpy as np


def last_station_index(mat: np.ndarray) -> np.ndarray:
    """행마다 값이 있는(NaN이 아닌) 마지막 열의 번호. 값이 하나도 없으면 -1.

    mat: 행 = 부품, 열 = 스테이션(번호순)의 시각.
    """
    has = ~np.isnan(mat)
    n_cols = mat.shape[1]
    last = n_cols - 1 - np.argmax(has[:, ::-1], axis=1)
    return np.where(has.any(axis=1), last, -1).astype(np.int64)


def group_kinds(group: np.ndarray, is_train: np.ndarray) -> np.ndarray:
    """묶음마다 구성: 'train'(train 부품만), 'test'(test 부품만), 'both'(걸침). 행 단위로 돌려준다."""
    group = np.asarray(group)
    is_train = np.asarray(is_train, dtype=bool)
    n_groups = int(group.max()) + 1 if len(group) else 0
    n_train = np.bincount(group, weights=is_train, minlength=n_groups)
    n_all = np.bincount(group, minlength=n_groups)
    kind = np.where(n_train == n_all, "train", np.where(n_train == 0, "test", "both"))
    return kind[group]


def share_recorded_before(count_before: np.ndarray, count_total: np.ndarray) -> np.ndarray:
    """부품마다 '시점까지 기록된 값의 수 / 전체 값의 수'. 전체가 0이면 NaN."""
    before = np.asarray(count_before, dtype=np.float64)
    total = np.asarray(count_total, dtype=np.float64)
    out = np.full(len(total), np.nan)
    ok = total > 0
    out[ok] = before[ok] / total[ok]
    return out


def ratio_outside(a: np.ndarray, b: np.ndarray, lo: float, hi: float, min_size: int) -> dict:
    """구간별 개수 a, b의 비(b / a)가 [lo, hi]를 벗어나는 구간 수. a가 min_size 미만인 구간은 빼고 센다."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    use = a >= min_size
    r = b[use] / a[use]
    return {"bins_compared": int(use.sum()), "bins_outside": int(((r < lo) | (r > hi)).sum()),
            "ratio_min": float(r.min()) if use.any() else None, "ratio_max": float(r.max()) if use.any() else None}
