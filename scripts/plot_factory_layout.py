"""공장 흐름 그림: 스테이션 52개를 흐름별로 놓은 도식 (3-3 방문 행렬, 3-4 시각 순서에서 추정).

비율은 DATA/interim/eda/station_visits_train.parquet(numeric 기준 방문)에서 센 값을 적어 넣은 것이다.
출력: DOCS/03_EDA/figures/3-3_factory_layout.png (Git 추적)
사용: .venv/bin/python scripts/plot_factory_layout.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
C = {"L0": "#2a78d6", "L1": "#eb6834", "L2": "#1baf7a", "L3": "#6250d6"}
TINT = {"L0": "#eaf2fc", "L1": "#fdf0ea", "L2": "#e8f7f1", "L3": "#efedfb"}

for f in ["/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf"]:
    if Path(f).exists():
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False})

BW, BH, GAP = 1.1, 0.72, 0.2  # 스테이션 상자 크기, 병렬 상자 사이 간격

fig, ax = plt.subplots(figsize=(25.5, 12.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 36)
ax.set_ylim(0, 17.6)
ax.axis("off")


def panel(x0, y0, x1, y1, line, title, share):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=TINT[line], ec=C[line], lw=1.6, zorder=1))
    ax.text(x0 + 0.3, y1 - 0.25, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK, zorder=5)
    ax.text(x0 + 0.3, y1 - 0.9, share, ha="left", va="top", fontsize=11.5, color=INK2, zorder=5)


def station(x, y, name, line, optional=False):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="white", ec=C[line] if not optional else MUTED, lw=1.5,
                                ls=(0, (3, 2)) if optional else "-", zorder=4))
    ax.text(x, y, name, ha="center", va="center", fontsize=11.5, color=INK if not optional else INK2, zorder=5)


def column(x, ymid, names, line, optional=False):
    """한 자리에 놓인 스테이션들. 2개 이상이면 '하나만 거침' 묶음으로 그린다. 좌우 끝 x를 돌려준다."""
    n = len(names)
    total = n * BH + (n - 1) * GAP
    top = ymid + total / 2 - BH / 2
    if n > 1:
        pad = 0.17
        ax.add_patch(FancyBboxPatch((x - BW / 2 - pad, ymid - total / 2 - pad), BW + 2 * pad, total + 2 * pad,
                                    boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec=MUTED, lw=1.0,
                                    alpha=0.9, zorder=3))
        label = "하나만" if not optional else "일부만"
        ax.text(x, ymid + total / 2 + pad + 0.12, label, ha="center", va="bottom", fontsize=9.5, color=INK2, zorder=5)
    elif optional:
        ax.text(x, ymid + BH / 2 + 0.12, "일부만", ha="center", va="bottom", fontsize=9.5, color=INK2, zorder=5)
    for i, nm in enumerate(names):
        station(x, top - i * (BH + GAP), nm, line, optional)
    ext = BW / 2 + (0.17 if n > 1 else 0)
    return x - ext, x + ext


def arrow(x0, y0, x1, y1, color=INK2, lw=1.6, style="-|>", ls="-", rad=0.0, z=2, ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def flow(x_start, dx, ymid, cols, line):
    """cols: [(이름 목록, optional)] 를 왼쪽에서 오른쪽으로 놓고 화살표로 잇는다."""
    prev_right = None
    for i, (names, opt) in enumerate(cols):
        left, right = column(x_start + i * dx, ymid, names, line, opt)
        if prev_right is not None:
            arrow(prev_right + 0.05, ymid, left - 0.05, ymid, color=C[line], lw=1.5, ms=11)
        prev_right = right
    return prev_right


# ---------- ① 진입 ----------
ax.text(0.5, 16.05, "① 진입: 세 곳 중 한 곳으로 들어온다", fontsize=14, fontweight="bold", color=INK, va="bottom")
panel(0.5, 11.3, 12.4, 15.9, "L0", "L0 앞쪽", "부품의 57%")
flow(1.5, 1.62, 12.95, [(["S0"], 0), (["S1"], 0), (["S2", "S3"], 0), (["S4", "S5"], 0), (["S6", "S7"], 0),
                        (["S8"], 0), (["S9", "S10", "S11"], 0)], "L0")
panel(0.5, 6.2, 12.4, 10.8, "L0", "L0 뒤쪽", "부품의 20%")
flow(1.5, 1.62, 7.85, [(["S12"], 0), (["S13"], 0), (["S14", "S15"], 0), (["S16", "S17"], 0), (["S18", "S19"], 0),
                       (["S20"], 0), (["S21", "S22", "S23"], 0)], "L0")
panel(0.5, 1.6, 12.4, 5.7, "L1", "L1", "부품의 23%")
column(11.22, 3.4, ["S24", "S25"], "L1")

# ---------- ② L2 ----------
ax.text(13.6, 16.05, "② L2: 일부만 거친다", fontsize=14, fontweight="bold", color=INK, va="bottom")
panel(13.6, 5.6, 17.0, 10.4, "L2", "L2", "부품의 30%")
column(15.3, 7.25, ["S26", "S27", "S28"], "L2")

# 진입 -> L3 직행(흐름마다 한 줄), 진입 -> L2. %는 모두 "그 흐름에서 출발한 부품 중" 비율이다
ax.text(13.6, 15.55, "전체 부품의 30%가 L2를 거치고,\n70%는 L2를 거치지 않고 바로 L3로 간다", ha="left", va="top",
        fontsize=11.5, color=INK, linespacing=1.5)
DIRECT = [(13.6, 2.6, "L0 앞쪽 부품의 86%는 바로 L3로"), (10.62, 1.7, "L0 뒤쪽 부품의 91%는 바로 L3로"),
          (2.4, 1.0, "L1 부품의 8%는 바로 L3로")]
for y, lw, text in DIRECT:
    ax.plot([12.4, 18.3], [y, y], color=INK2, lw=lw, zorder=2, solid_capstyle="butt")
    ax.text(15.3, y + 0.13, text, ha="center", va="bottom", fontsize=10.5, color=INK)
arrow(12.4, 12.3, 13.6, 9.6, color=C["L2"], lw=1.4, ms=12, rad=-0.15)
arrow(12.4, 8.3, 13.6, 8.3, color=C["L2"], lw=1.2, ms=12)
arrow(12.4, 3.7, 13.6, 6.4, color=C["L2"], lw=3.0, ms=16, rad=0.15)
ax.text(12.55, 12.45, "L0 앞쪽\n부품의 14%", ha="left", va="bottom", fontsize=9.5, color=INK2, linespacing=1.3)
ax.text(12.5, 8.42, "L0 뒤쪽\n부품의 9%", ha="left", va="bottom", fontsize=9.5, color=INK2, linespacing=1.3)
ax.text(12.55, 3.5, "L1\n부품의 92%", ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.3)
ax.text(15.6, 5.3, "초록 화살표의 %: 그 흐름의\n부품 중 L2를 거치는 비율\nS28은 부품의 0.8%만 거친다", ha="center", va="top",
        fontsize=10.5, color=INK2, linespacing=1.5)
arrow(17.0, 8.0, 18.3, 8.0, color=C["L2"], lw=1.6, ms=13)

# ---------- ③ L3 ----------
ax.text(19.2, 16.05, "③ L3: 두 흐름 중 하나를 거친다 (거의 모든 부품이 L3를 거침)", fontsize=14, fontweight="bold",
        color=INK, va="bottom")
# 분기 기둥
ax.plot([18.3, 18.3], [2.4, 13.6], color=INK2, lw=2.2, zorder=2, solid_capstyle="round")
arrow(18.3, 12.75, 19.2, 12.75, color=INK2, lw=2.6, ms=18)
arrow(18.3, 3.9, 19.2, 3.9, color=INK2, lw=1.2, ms=12)
panel(19.2, 10.4, 35.6, 15.9, "L3", "L3 주 흐름", "부품의 94.6%")
flow(20.4, 1.95, 12.75, [(["S29"], 0), (["S30"], 0), (["S31", "S32"], 1), (["S33"], 0), (["S34"], 0),
                          (["S35", "S36"], 0), (["S37"], 0), (["S38"], 1)], "L3")
panel(19.2, 1.6, 35.6, 6.8, "L3", "L3 다른 흐름", "부품의 5.1%")
flow(20.05, 1.45, 4.1, [(["S39"], 0), (["S40"], 0), (["S41"], 0), (["S42"], 1), (["S43", "S44"], 0), (["S45"], 0),
                        (["S46"], 1), (["S47"], 0),
                        (["S48"], 0), (["S49", "S50"], 0), (["S51"], 0)], "L3")
ax.text(27.4, 9.95, "S31·S32는 주 흐름 부품의 5%, S38은 2%만 거친다 (S38은 주 흐름에만 있다)", ha="center", va="top", fontsize=10.5, color=INK2)
ax.text(27.4, 8.6, "두 흐름을 모두 거친 부품은 0개", ha="center", va="center", fontsize=11.5, color=INK)

ax.text(27.4, 2.15, "S42는 부품 15개, S46은 부품 1개만 거친다", ha="center", va="center", fontsize=10.5, color=INK2)

# ---------- 범례 ----------
lx, ly = 0.9, 0.7
station(lx + 0.55, ly, "S0", "L0")
ax.text(lx + 1.3, ly, "그 흐름의 부품이 모두 거침", va="center", fontsize=11, color=INK2)
gx = lx + 7.0
ax.add_patch(FancyBboxPatch((gx - 0.1, ly - BH / 2 - 0.1), 2 * BW + 0.4, BH + 0.2, boxstyle="round,pad=0,rounding_size=0.15",
                            fc="white", ec=MUTED, lw=1.0, zorder=3))
station(gx + BW / 2, ly, "S2", "L0")
station(gx + BW * 1.5 + 0.2, ly, "S3", "L0")
ax.text(gx + 2 * BW + 0.6, ly, "묶음 안에서 하나만 거침(같은 작업을 하는 설비로 추정)", va="center", fontsize=11, color=INK2)
station(lx + 21.0, ly, "S38", "L3", optional=True)
ax.text(lx + 21.75, ly, "일부 부품만 거침", va="center", fontsize=11, color=INK2)

fig.text(0.012, 0.985, "부품이 거치는 공장 흐름 (방문 기록에서 추정)", ha="left", va="top", fontsize=19, fontweight="bold", color=INK)
fig.text(0.012, 0.945, "train 부품 1,183,747개 기준. date 파일에 있는 스테이션 52개를 모두 그렸다. "
         "스테이션은 번호순이고, 이 순서가 실제로 거치는 순서라는 것은 3-4에서 시각으로 확인했다.", ha="left", va="top", fontsize=11.5, color=INK2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.915, bottom=0.01)
out = Path("DOCS/03_EDA/figures/3-3_factory_layout.png")
fig.savefig(out, dpi=110, facecolor=SURFACE)
print("saved", out)
