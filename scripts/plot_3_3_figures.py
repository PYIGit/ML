"""3-3 그림 5개 생성 (설계: DOCS/03_EDA/3-3_MISSINGNESS_AND_ROUTES.md 컨펌 #3 = A).

입력: DATA/interim/eda/ 의 3-3 산출물 (missing_rate_summary.csv, nonnull_per_row_dist.csv,
      station_visit_rate.csv, routes_*.parquet, line_combinations.csv)
출력: DOCS/03_EDA/figures/3-3_*.png (Git 추적)
색: dataviz 스킬의 기본 팔레트 슬롯 1~3 (라이트, `--pairs all` 검증 통과. 아쿠아는 대비 3:1 미만이라
    마커 모양, 직접 라벨, 범례로 색 외 구분을 함께 쓴다).
사용: .venv/bin/python scripts/plot_3_3_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

EDA = Path("DATA/interim/eda")
FIG = Path("DOCS/03_EDA/figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = {"numeric": "#2a78d6", "date": "#eb6834", "categorical": "#1baf7a"}
MARK = {"numeric": "o", "date": "s", "categorical": "^"}
KINDS = list(SERIES)
KIND_KO = {"numeric": "numeric", "date": "date", "categorical": "categorical"}

for f in ["/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf"]:
    if Path(f).exists():
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "figure.dpi": 100,
})


def finish(fig, ax_list, name, title, subtitle=None):
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    for ax in ax_list:
        ax.set_axisbelow(True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / name, flush=True)


def fig_missing_rate():
    d = pl.read_csv(EDA / "missing_rate_summary.csv")
    bins = d["bin"].unique(maintain_order=True).to_list()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    h, gap = 0.26, 0.02
    y = np.arange(len(bins))
    for i, k in enumerate(KINDS):
        sub = d.filter(pl.col("data_kind") == k)
        tot = sub["n_columns"].sum()
        share = sub["n_columns"].to_numpy() / tot * 100
        pos = y + (i - 1) * (h + gap)
        ax.barh(pos, share, height=h, color=SERIES[k], label=f"{k} (컬럼 {tot:,}개)")
        for p, s, n in zip(pos, share, sub["n_columns"]):
            ax.text(s + 0.8, p, f"{s:.1f}% ({n:,})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, bins)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("해당 구간의 컬럼 비율 (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.subplots_adjust(top=0.82)
    finish(fig, [ax], "3-3_column_missing_rate.png", "컬럼 결측률 구간별 컬럼 비율 (train)",
           "결측률 = 컬럼의 빈 값 비율. numeric과 date는 컬럼 대부분이 90% 이상 비어 있다")


def fig_nonnull_per_row():
    d = pl.read_csv(EDA / "nonnull_per_row_dist.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
    for ax, k in zip(axes, KINDS):
        sub = d.filter(pl.col("data_kind") == k)
        x, c = sub["n_nonnull"].to_numpy(), sub["count"].to_numpy()
        ax.bar(x, c, width=1.8, color=SERIES[k], linewidth=0)
        mean = float((x * c).sum() / c.sum())
        ax.axvline(mean, color=INK2, lw=1, ls="--")
        ax.text(mean, ax.get_ylim()[1] * 0.96, f"평균 {mean:.1f} ", fontsize=8.5, color=INK2, va="top", ha="right")
        n0 = int(c[x == 0].sum()) if (x == 0).any() else 0
        ax.set_title(f"{k}  (값이 0개인 행 {n0:,})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("행마다 값이 있는 컬럼 수")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k" if v else "0")
    axes[0].set_ylabel("부품 수")
    fig.subplots_adjust(top=0.78, wspace=0.28)
    finish(fig, axes, "3-3_nonnull_per_row.png", "행별 값이 있는 컬럼 수 분포 (train, 1,183,747행)",
           "categorical은 값이 하나도 없는 행이 많아 0에 막대가 크게 선다")


def fig_station_visit_rate():
    d = pl.read_csv(EDA / "station_visit_rate.csv")
    num = d.filter(pl.col("data_kind") == "numeric")
    order = (d.select("station_key", "line").unique()
             .with_columns(pl.col("station_key").str.extract(r"_S(\d+)", 1).cast(pl.Int32).alias("s"))
             .sort(["line", "s"]))
    keys = order["station_key"].to_list()
    pos = {k: i for i, k in enumerate(keys)}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k in KINDS:
        sub = d.filter(pl.col("data_kind") == k)
        xs = [pos[s] for s in sub["station_key"]]
        ax.scatter(xs, sub["visit_rate"] * 100, s=34 if k != "date" else 26, marker=MARK[k],
                   facecolor=SERIES[k] if k != "date" else "none",
                   edgecolor=SERIES[k] if k == "date" else SURFACE, linewidth=1.3 if k == "date" else 0.8,
                   label=f"{k} ({sub.height}개 스테이션)", zorder=3 if k != "date" else 4)
    lines = order["line"].to_list()
    start = 0
    for ln in range(4):
        idx = [i for i, l in enumerate(lines) if l == ln]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        if ln > 0:
            ax.axvline(lo - 0.5, color=GRID, lw=1.2)
        ax.text((lo + hi) / 2, 104, f"L{ln}", ha="center", fontsize=9.5, color=INK2, fontweight="bold")
    ax.set_xticks(range(len(keys)), [k.split("_")[1] for k in keys], rotation=90, fontsize=7.5)
    ax.set_xlim(-0.8, len(keys) - 0.2)
    ax.set_ylim(-3, 108)
    ax.set_ylabel("방문율 (%)")
    ax.set_xlabel("스테이션 (라인별로 번호 순)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    fig.subplots_adjust(top=0.84)
    finish(fig, [ax], "3-3_station_visit_rate.png", "스테이션별 방문율: 기준 파일 3가지 비교 (train)",
           "방문 = 그 스테이션 컬럼 중 하나라도 값이 있음. numeric과 date는 거의 겹치고 categorical은 훨씬 낮다")


def fig_route_cumulative():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    styles = {"numeric": "-", "date": (0, (4, 3)), "categorical": "-"}
    end = {}
    for k in KINDS:
        t = pl.read_parquet(EDA / f"routes_{k}.parquet")
        n = t["n"].to_numpy()
        cum = np.cumsum(n) / n.sum() * 100
        rank = np.arange(1, len(n) + 1)
        ax.plot(rank, cum, color=SERIES[k], lw=2, ls=styles[k], label=f"{k} (경로 {len(n):,}개)")
        end[k] = (len(n), cum[-1])
    ax.set_xscale("log")
    ax.set_ylim(0, 102)
    ax.set_xlabel("경로 순위 (부품 수가 많은 순, 로그 축)")
    ax.set_ylabel("누적 점유율 (%)")
    for r in (10, 100, 1000):
        ax.axvline(r, color=GRID, lw=1, zorder=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.subplots_adjust(top=0.82)
    finish(fig, [ax], "3-3_route_cumulative_share.png", "상위 N개 경로의 누적 점유율 (train)",
           "경로 = 방문 스테이션 집합. numeric과 date의 곡선은 거의 같다(점선이 실선과 겹침)")


def fig_line_combinations():
    d = pl.read_csv(EDA / "line_combinations.csv").sort("n", descending=True)
    labels = d["combo"].to_list()
    y = np.arange(len(labels))[::-1]
    small = d["small_group"].to_numpy()
    col = np.where(small, MUTED, SERIES["numeric"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, gridspec_kw={"wspace": 0.06})
    a1.barh(y, d["n"], height=0.62, color=col)
    a1.set_xscale("log")
    a1.set_xlim(0.7, 6e6)
    for yy, n in zip(y, d["n"]):
        a1.text(n * 1.25, yy, f"{n:,}", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(y, labels)
    a1.set_xlabel("부품 수 (로그 축)")
    a1.grid(axis="y", visible=False)
    r = d["rate"].to_numpy() * 100
    lo, hi = d["ci_low"].to_numpy() * 100, d["ci_high"].to_numpy() * 100
    XMAX = 2.0
    for yy, rr, l, h, s_, n, k in zip(y, r, lo, hi, small, d["n"], d["k"]):
        c = MUTED if s_ else SERIES["numeric"]
        a2.plot([l, min(h, XMAX)], [yy, yy], color=c, lw=2, solid_capstyle="round", zorder=2)
        if h > XMAX:
            a2.text(XMAX, yy, "▸", color=c, fontsize=9, va="center", ha="center", zorder=3)
        if rr <= XMAX:
            a2.scatter([rr], [yy], s=46, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        tag = f"{rr:.2f}% (불량 {k:,}" + (", 소표본)" if s_ else ")")
        a2.text(1.02, yy, tag, va="center", fontsize=8.5, color=INK2, transform=a2.get_yaxis_transform())
    a2.axvline(0.581, color=INK2, lw=1, ls="--", zorder=1)
    a2.text(0.581, len(labels) - 0.35, " 전체 0.581%", fontsize=8.5, color=INK2, ha="left")
    a2.set_xlim(0, XMAX)
    a2.set_xlabel("불량률 % (95% Wilson 구간. ▸는 구간이 2%를 넘음)")
    a2.grid(axis="y", visible=False)
    a1.set_title("라인 조합별 부품 수", loc="left", fontsize=10, color=INK)
    a2.set_title("라인 조합별 불량률", loc="left", fontsize=10, color=INK)
    fig.subplots_adjust(top=0.84)
    finish(fig, [a1, a2], "3-3_line_combinations.png", "라인 방문 조합별 부품 수와 불량률 (numeric 기준, train)",
           "회색 = 부품 수 1,000 미만(소표본). 라인 방문 = 그 라인 스테이션 중 하나라도 방문. none은 numeric 값이 없는 582개")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_missing_rate()
    fig_nonnull_per_row()
    fig_station_visit_rate()
    fig_route_cumulative()
    fig_line_combinations()
