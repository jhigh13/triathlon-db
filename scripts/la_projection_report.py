#!/usr/bin/env python
"""
la_projection_report.py -- Branded PDF of the "if LA were today" medal projection.

Reads the projection CSVs written by
    predict_program.py --virtual_olympic --gender {men,women} --ranking_prior 0.5
(outputs/projection_LA_{men,women}.csv) and produces a shareable PDF:

    * Cover with methodology.
    * A "Top 10 medal contenders" zoom per gender (P(medal) bar chart, USA in red).
    * The full 60-athlete tables as an appendix.

Usage:
    python scripts/la_projection_report.py
    python scripts/la_projection_report.py --output "ppt files/LA_projection.pdf"
"""
import argparse
import os
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGO = os.path.join(_REPO, "docs", "power_bi_files", "USA_Triathlon_Logo.jpg")

# USAT brand
NAVY = "#002060"
RED = "#C00000"
LIGHT = "#EEF1F7"
GRAY = "#606060"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

PAGE_W, PAGE_H = 11.0, 8.5  # landscape letter

COUNTRY_SHORT = {
    "Individual Neutral Athlete": "Neutral (AIN)",
    "United States": "United States",
    "Great Britain": "Great Britain",
}


def _short_country(c: str) -> str:
    return COUNTRY_SHORT.get(str(c), str(c))


def _header(fig, title: str, subtitle: str = ""):
    """Navy title band + red accent + logo, drawn in figure coordinates."""
    fig.patches.append(Rectangle((0, 0.90), 1.0, 0.10, transform=fig.transFigure,
                                 color=NAVY, zorder=0))
    fig.patches.append(Rectangle((0, 0.888), 1.0, 0.012, transform=fig.transFigure,
                                 color=RED, zorder=0))
    fig.text(0.035, 0.945, title, color="white", fontsize=22, fontweight="bold", va="center")
    if subtitle:
        fig.text(0.035, 0.912, subtitle, color="#AABBDD", fontsize=10.5, va="center")
    if os.path.exists(LOGO):
        ax = fig.add_axes([0.88, 0.905, 0.10, 0.085], anchor="NE", zorder=5)
        ax.imshow(plt.imread(LOGO)); ax.axis("off")


def _footer(fig, note: str):
    fig.patches.append(Rectangle((0, 0), 1.0, 0.028, transform=fig.transFigure, color=NAVY, zorder=0))
    fig.text(0.035, 0.045, note, color=GRAY, fontsize=7.2, style="italic", va="center")


def cover_page(pdf, gen_date):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="white", zorder=-1))
    fig.patches.append(Rectangle((0, 0.62), 1.0, 0.06, transform=fig.transFigure, color=RED, zorder=0))
    if os.path.exists(LOGO):
        ax = fig.add_axes([0.42, 0.70, 0.16, 0.16], anchor="C"); ax.imshow(plt.imread(LOGO)); ax.axis("off")
    fig.text(0.5, 0.565, "LA28 Olympic Triathlon", color=NAVY, fontsize=34,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.505, '"If the Race Were Today" — Medal Projection', color=NAVY,
             fontsize=20, ha="center")
    fig.text(0.5, 0.45, f"Men & Women  ·  Generated {gen_date:%B %d, %Y}", color=GRAY,
             fontsize=12, ha="center")
    import textwrap
    paras = [
        "How to read this: We reconstruct the Olympic field from current World Rankings under the real "
        "quota (60 athletes, max 3 per country, a 3rd only if inside the top 30), removing retired athletes. "
        "Each athlete's finish is projected by our race model, anchored by athlete strength (Elo) and world "
        "ranking, then simulated thousands of times to estimate the probability of winning a medal (top 3).",
        "P(medal) = share of simulations in which the athlete finishes on the podium. These reflect current "
        "form: athletes returning from a lighter season rank lower than their reputation.",
    ]
    blurb = "\n\n".join(textwrap.fill(p, width=95) for p in paras)
    fig.text(0.5, 0.29, blurb, color="#333333", fontsize=11.5, ha="center", va="center", linespacing=1.5)
    _footer(fig, "Source: World Triathlon results & rankings database  ·  USA Triathlon High Performance")
    pdf.savefig(fig); plt.close(fig)


def top10_page(pdf, gender: str, df: pd.DataFrame, gen_date):
    top = df.head(10).iloc[::-1]  # best on top of a barh
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _header(fig, f"{gender} — Top 10 Medal Contenders",
            'Chance of a medal (top 3) and a win, "if the race were today"')

    ax = fig.add_axes([0.30, 0.11, 0.60, 0.72])
    y = list(range(len(top)))
    h = 0.38
    ax.barh([i + h / 2 for i in y], top["p_medal"], height=h, color=NAVY,
            edgecolor="white", label="Medal (top 3)")
    ax.barh([i - h / 2 for i in y], top["p_win"], height=h, color=RED,
            edgecolor="white", label="Win")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.athlete_full_name}\n{_short_country(r.country)}" for r in top.itertuples()],
                       fontsize=9.5)
    for tick, r in zip(ax.get_yticklabels(), top.itertuples()):
        if r.country == "United States":
            tick.set_color(RED); tick.set_fontweight("bold")

    xmax = max(top["p_medal"].max() * 1.16, 10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlabel("Probability (%)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    for i, r in enumerate(top.itertuples()):
        ax.text(r.p_medal + xmax * 0.012, i + h / 2, f"{r.p_medal:.0f}%", va="center",
                fontsize=8.5, fontweight="bold", color=NAVY)
        ax.text(r.p_win + xmax * 0.012, i - h / 2, f"{r.p_win:.0f}%", va="center",
                fontsize=8.5, fontweight="bold", color=RED)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    # Rank chips down the left margin
    for i, r in enumerate(top.itertuples()):
        fig.text(0.055, 0.11 + 0.72 * (i + 0.5) / len(top), f"#{int(r.proj)}",
                 color=NAVY, fontsize=15, fontweight="bold", va="center", ha="center")

    fav = df.iloc[0]
    fig.text(0.30, 0.865,
             f"Projected favorite: {fav.athlete_full_name} ({_short_country(fav.country)}) — "
             f"{fav.p_medal:.0f}% medal, {fav.p_win:.0f}% win.   USA names in red.",
             color=RED, fontsize=10.5, fontweight="bold", va="center")
    _footer(fig, f"Generated {gen_date:%b %d, %Y}  ·  P(medal)=share of simulations finishing top 3  ·  "
                 "Field: 60 by world ranking, max 3/country.")
    pdf.savefig(fig); plt.close(fig)


def appendix_table_page(pdf, gender: str, df: pd.DataFrame, gen_date):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _header(fig, f"Appendix — {gender}: Full Field",
            "All qualified athletes, projected finish order")

    cols = ["#", "Athlete", "Country", "WRank", "Win%", "Medal%"]

    def _rows(sub):
        out = []
        for r in sub.itertuples():
            out.append([int(r.proj), str(r.athlete_full_name), _short_country(r.country),
                        int(r.rank_position) if pd.notna(r.rank_position) else "-",
                        f"{r.p_win:.1f}", f"{r.p_medal:.1f}"])
        return out

    half = (len(df) + 1) // 2
    left, right = df.iloc[:half], df.iloc[half:]
    for k, (sub, x0) in enumerate([(left, 0.035), (right, 0.52)]):
        ax = fig.add_axes([x0, 0.05, 0.445, 0.80]); ax.axis("off")
        tbl = ax.table(cellText=_rows(sub), colLabels=cols, loc="center",
                       cellLoc="center", colWidths=[0.09, 0.44, 0.24, 0.1, 0.11, 0.12])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.2); tbl.scale(1, 1.18)
        for (rr, cc), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D9D9D9")
            if cc == 1:  # left-align the Athlete column
                cell.set_text_props(ha="left"); cell._loc = "left"
            if rr == 0:
                cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
                continue
            is_usa = sub.iloc[rr - 1]["country"] == "United States"
            cell.set_facecolor("#FBE9E9" if is_usa else ("white" if rr % 2 else LIGHT))
            if is_usa:
                cell.set_text_props(color=RED, fontweight="bold")

    _footer(fig, f"Generated {gen_date:%b %d, %Y}  ·  WRank = current World Ranking position  ·  "
                 "USA highlighted.  Probabilities from prior-anchored simulation.")
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--women_csv", default=os.path.join(_REPO, "outputs", "projection_LA_women.csv"))
    ap.add_argument("--men_csv", default=os.path.join(_REPO, "outputs", "projection_LA_men.csv"))
    ap.add_argument("--output", default=os.path.join(_REPO, "ppt files", "LA_medal_projection.pdf"))
    args = ap.parse_args()

    gen_date = date.today()
    women = pd.read_csv(args.women_csv)
    men = pd.read_csv(args.men_csv)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with PdfPages(args.output) as pdf:
        cover_page(pdf, gen_date)
        top10_page(pdf, "Men", men, gen_date)
        top10_page(pdf, "Women", women, gen_date)
        appendix_table_page(pdf, "Men", men, gen_date)
        appendix_table_page(pdf, "Women", women, gen_date)
    print(f"Saved PDF: {args.output}")


if __name__ == "__main__":
    main()
