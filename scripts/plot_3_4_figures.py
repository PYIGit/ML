"""3-4 그림 2개 생성 (설계: DOCS/03_EDA/3-4_TIME_STRUCTURE.md).

입력: DATA/interim/eda/ 의 3-4 산출물 (station_transitions.csv, time_bin_summary.csv)
출력: DOCS/03_EDA/figures/3-4_*.png (Git 추적)
색: dataviz 스킬 기본 팔레트 슬롯 1(파랑), 2(주황). 두 계열은 마커 모양과 범례로도 구분한다.
사용: .venv/bin/python scripts/plot_3_4_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, LogNorm

EDA = Path("DATA/interim/eda")
FIG = Path("DOCS/03_EDA/figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

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
        fig.text(0.01, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    for ax in ax_list:
        ax.set_axisbelow(True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / name, flush=True)


def station_no(key: str) -> int:
    return int(key.split("_S")[1])


def fig_station_flow():
    t = pl.read_csv(EDA / "station_transitions.csv")
    stations = sorted(set(t["from_station"]) | set(t["to_station"]), key=station_no)
    idx = {s: i for i, s in enumerate(stations)}
    m = np.full((len(stations), len(stations)), np.nan)
    for r in t.iter_rows(named=True):
        m[idx[r["from_station"]], idx[r["to_station"]]] = r["n"]
    cmap = LinearSegmentedColormap.from_list("blue", ["#dbe9fa", BLUE, "#0d3a75"])
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(m, cmap=cmap, norm=LogNorm(vmin=100, vmax=np.nanmax(m)), aspect="equal")
    ax.set_xticks(range(len(stations)), [s.split("_")[1] for s in stations], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(stations)), [s.split("_")[1] for s in stations], fontsize=7.5)
    ax.grid(False)
    lines = [int(s[1]) for s in stations]
    for ln in sorted(set(lines)):
        pos = [i for i, x in enumerate(lines) if x == ln]
        lo, hi = pos[0] - 0.5, pos[-1] + 0.5
        for a, b in ((lo, lo), (hi, hi)):
            ax.axhline(a, color=MUTED, lw=0.8)
            ax.axvline(b, color=MUTED, lw=0.8)
        ax.text((lo + hi) / 2, -1.6, f"L{ln}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK2)
        ax.text(len(stations) + 0.3, (lo + hi) / 2, f"L{ln}", ha="left", va="center", fontsize=10, fontweight="bold",
                color=INK2)
    ax.set_xlabel("뒤 스테이션 (다음에 거친 곳)")
    ax.set_ylabel("앞 스테이션")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.06)
    cb.set_label("이어서 거친 부품 수 (로그 색)", color=INK2)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.9)
    finish(fig, [ax], "3-4_station_flow.png", "부품이 이어서 거친 스테이션 쌍 (train, 시각순)",
           "칸 = 앞 스테이션 다음에 뒤 스테이션을 거친 부품 수. 대각선 아래(번호 역순)에는 칸이 하나도 없다. 부품 100개 미만인 칸은 가장 옅은 색")


def fig_time_trend():
    t = pl.read_csv(EDA / "time_bin_summary.csv")
    x = t["t_start"].to_numpy() + 5
    n = t["n"].to_numpy()
    rate = t["rate"].to_numpy() * 100
    lo, hi = t["sim_lo"].to_numpy() / n * 100, t["sim_hi"].to_numpy() / n * 100
    high = (t["outside"] == "high").to_numpy()
    small = t["small_group"].to_numpy()
    overall = t["k"].sum() / t["n"].sum() * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.7], "hspace": 0.12})
    a1.bar(x, n, width=8, color=BLUE)
    a1.set_ylabel("부품 수")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    a1.grid(axis="x", visible=False)
    a1.set_title("시간 구간(폭 10)별 부품 수", loc="left", fontsize=10, color=INK)
    lo, hi = np.where(small, np.nan, lo), np.where(small, np.nan, hi)  # 소표본 구간은 범위가 너무 넓어 그리지 않는다
    a2.fill_between(x, lo, hi, step="mid", color=GRID, zorder=1,
                    label="불량이 시기와 무관하게 놓였을 때의 95% 범위")
    a2.axhline(overall, color=INK2, lw=1, ls="--", zorder=2)
    a2.text(x.min() - 8, overall + 0.04, f"전체 {overall:.2f}%", fontsize=8.5, color=INK2, va="bottom", ha="left")
    ok = ~small
    a2.scatter(x[ok & ~high], rate[ok & ~high], s=18, color=BLUE, zorder=3, label="구간 불량률")
    a2.scatter(x[ok & high], rate[ok & high], s=40, color=ORANGE, marker="D", edgecolor=SURFACE, linewidth=0.8, zorder=4,
               label="범위보다 높은 구간")
    a2.scatter(x[small], np.minimum(rate[small], 3.0), s=18, color=MUTED, marker="x", zorder=3,
               label="부품 1,000개 미만 구간")
    a2.set_ylim(0, 3.0)
    a2.set_ylabel("불량률 (%)")
    a2.set_xlabel("부품의 최초 시각 (date 값)")
    a2.set_title("시간 구간별 불량률", loc="left", fontsize=10, color=INK)
    a2.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=2)
    fig.subplots_adjust(top=0.89)
    finish(fig, [a1, a2], "3-4_time_volume_defect_rate.png", "시기별 생산량과 불량률 (train)",
           "불량률이 회색 범위 위에 있는 구간(주황)이 앞쪽 절반에 몰려 있다. 무작위 배치 5,000번과 비교")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_station_flow()
    fig_time_trend()
