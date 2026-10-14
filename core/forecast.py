#!/usr/bin/env python3
"""Forecast ledger: score every researched estimate, not just bets.

PROTECTED CORE — the trading agent must not edit files under core/.
This is the only writer of journal/forecasts.jsonl (resolve.py settles rows).

The experiment's honest metric is brier_delta — is the agent's probability a
better forecast than the market's own price? — and measuring that needs no
stake. Research that ends in a skip (no-edge, market-agrees) still produced
an estimate; recording it here turns ~10-30 researched candidates/day into
scored calibration feedback instead of ~0-1 settled bets/day.

No caps, no edge floor, no fill: the market baseline is the MID at record
time ((bid+ask)/2), not the ask a bet would fill at — a forecast has no
transaction, and the mid is the stricter benchmark. Forecast brier_delta is
therefore NOT comparable to bet brier_delta; score.py reports them in
separate sections. est_prob is the agent's honest belief, formed before
anchoring on the price, exactly as for bets.

One live forecast per market+outcome: recurring re-checks of the same market
must not flood the stats with correlated rows. A materially changed read
(|delta est_prob| >= 0.05, or a changed funnel decision) may replace the live
row via record --supersede: the rows are linked (supersedes / superseded_by),
only the latest row counts in headline scoring, and the superseded row still
settles into score.py's separate revised-away slice - whether revisions
actually improve estimates is measured, not assumed (PLBY 2026-08-10: the
revision was worse than the original).

Venues (operator, 2026-10-14): --venue polymarket (the default) reads gamma
and the CLOB and writes journal/forecasts.jsonl, exactly as before. --venue
myriad takes a Myriad market slug as --market-id, applies core/myriad.py's
protected filter (order book, real-money collateral, binary, no sports), reads
the outcome's Myriad order book for the mid, and writes
journal/forecasts-myriad.jsonl. The two files are never pooled: the Polymarket
history stays a closed era, and every reader of forecasts.jsonl keeps seeing
only Polymarket rows.

Usage:
  record: python3 core/forecast.py record [--venue myriad] --market-id 123 \
            --outcome Yes --est-prob 0.62 --category econ --skip-reason no-edge \
            [--fit-score 4] [--note "..."] [--strategy-rev abc1234] \
            [--confirm-extreme] [--supersede]
  status: python3 core/forecast.py status [--venue myriad]
"""
import argparse
import datetime as dt
import json
import pathlib
import sys
import uuid
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import myriad  # noqa: E402
import pmapi  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "journal" / "forecasts.jsonl"
FORECASTS_MYRIAD = ROOT / "journal" / "forecasts-myriad.jsonl"


def forecasts_path(venue):
    return FORECASTS_MYRIAD if venue == "myriad" else FORECASTS

# A supersede must change something material: a re-record of an unchanged
# estimate is the correlated-row flooding the one-live-row rule exists to
# prevent, not a revision.
MIN_REVISION_DELTA = 0.05

# An |est_prob - mid| gap this large is either a deliberate extreme
# disagreement (rare: the outside-view-veto class) or an inverted outcome
# side (fe954ed9f325: --outcome No with the Yes-side est_prob, silently
# recording the opposite of the researched belief). The flag costs one
# keystroke exactly when the agent should be pausing anyway; the typo gets
# caught at the only moment it is fixable.
EXTREME_DISAGREEMENT = 0.40


def read_forecasts(path=None):
    path = path or FORECASTS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_status(rows):
    print(json.dumps({
        "total": len(rows),
        "by_status": dict(Counter(r["status"] for r in rows)),
        "by_skip_reason": dict(Counter(r.get("skip_reason") or "?" for r in rows)),
        "settled_wins": sum(1 for r in rows if r["status"] == "won"),
        "revised_away": sum(1 for r in rows if r.get("superseded_by")),
    }, indent=2))


def _num(value):
    """gamma numeric field -> float rounded to cents, or None if absent/unparseable."""
    if value in (None, ""):
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def polymarket_quote(args):
    """Market fields for the row, plus (bid, ask) of the named outcome."""
    m = pmapi.gamma_market(args.market_id)
    if m.get("closed"):
        sys.exit("REJECTED: market is closed")
    tokens = pmapi.market_tokens(m)
    if tokens is None:
        sys.exit(f"REJECTED: no usable CLOB token ids (Gamma version "
                 f"{pmapi.market_version(m)!r}; supported: "
                 f"{sorted(pmapi.SUPPORTED_VERSIONS)})")
    if args.outcome not in tokens:
        sys.exit(f"REJECTED: outcome {args.outcome!r} not in {list(tokens)}")
    bid, ask = pmapi.best_prices(tokens[args.outcome])
    return {
        "question": m.get("question"),
        "slug": m.get("slug"),
        "end_date": m.get("endDate"),
        "token_id": tokens[args.outcome],
        # Book size at record time, from the same gamma record scan.py reads.
        # Added 2026-09-09 so research edge can be split by liquidity later;
        # gnhf run 5 had to drop that feature for lack of it. None when gamma
        # omits the field - never a guess, never a later re-read.
        "liquidity_at_record": _num(m.get("liquidityNum")),
        "volume_24h_at_record": _num(m.get("volume24hr")),
    }, bid, ask


def myriad_quote(args):
    """Same contract as polymarket_quote, from Myriad's market and order book.
    The protected venue filter runs here too, so a slug passed directly cannot
    reach a market the scan would never offer (sports, AMM, play money)."""
    if not myriad.venue_config().get("forecasts_enabled"):
        sys.exit("REJECTED: venues.myriad.forecasts_enabled is false in "
                 "config/protected.json")
    try:
        m = myriad.market(args.market_id)
    except myriad.Blocked as e:
        sys.exit(f"REJECTED: {e}")
    why = myriad.inadmissible(m, dt.datetime.now(dt.timezone.utc))
    if why:
        sys.exit(f"REJECTED: Myriad market not admissible: {why}")
    titles = myriad.outcome_titles(m)
    if args.outcome not in titles:
        sys.exit(f"REJECTED: outcome {args.outcome!r} not in {titles}")
    # Myriad addresses an outcome by its id (the order book's ?outcome= and
    # resolvedOutcomeId), not by its position in the list.
    idx = m["outcomes"][titles.index(args.outcome)]["id"]
    bid, ask = myriad.best_prices(m["slug"], idx)
    # Myriad's books are thin and often one-sided; a lone quote is not a mid
    # (it would pull the benchmark toward the empty side), so require both.
    if bid is None or ask is None:
        sys.exit(f"REJECTED: one-sided Myriad book (bid={bid} ask={ask}) - no "
                 f"two-sided mid to benchmark against")
    return {
        "venue": "myriad",
        "question": m.get("title"),
        "slug": m.get("slug"),
        "end_date": myriad.iso_z(m.get("expiresAt")),
        "token_id": myriad.token_id(m, idx),
        "liquidity_at_record": _num(m.get("liquidity")),
        "volume_24h_at_record": _num(m.get("volume24h")),
        # Per-market taker fee, recorded for phase B's fill model.
        "taker_fee_bps_at_record": (m.get("fees") or {}).get("taker_fee_bps"),
        "topics": m.get("topics") or [],
    }, bid, ask


def cmd_record(args, rows):
    if not 0.0 < args.est_prob < 1.0:
        sys.exit("REJECTED: est-prob must be in (0,1)")
    live = [r for r in rows
            if r["market_id"] == args.market_id and r["outcome"] == args.outcome
            and r["status"] == "open" and not r.get("superseded_by")]
    if live and not args.supersede:
        sys.exit("REJECTED: already have an open forecast on this market+outcome "
                 "(a materially changed read may supersede it: --supersede)")
    old = None
    if args.supersede:
        if not live:
            sys.exit("REJECTED: --supersede, but no live open forecast on this "
                     "market+outcome to supersede")
        old = live[0]
        # round like ledger.py's edge field so an exactly-boundary revision
        # (0.33 - 0.28 = 0.049999...) does not float-drop below the gate
        if (round(abs(args.est_prob - old["est_prob"]), 4) < MIN_REVISION_DELTA
                and args.skip_reason == old.get("skip_reason")):
            sys.exit(f"REJECTED: supersede needs a material change — "
                     f"|delta est_prob| >= {MIN_REVISION_DELTA} "
                     f"(old {old['est_prob']}, new {args.est_prob}) or a changed "
                     f"skip-reason (old {old.get('skip_reason')!r})")

    info, bid, ask = (myriad_quote(args) if args.venue == "myriad"
                      else polymarket_quote(args))
    if bid is None and ask is None:
        sys.exit("REJECTED: empty book — no market probability to benchmark against")
    mid = (bid + ask) / 2 if bid is not None and ask is not None else bid or ask

    gap = abs(args.est_prob - mid)
    if gap > EXTREME_DISAGREEMENT and not args.confirm_extreme:
        sys.exit(f"REJECTED: est_prob {args.est_prob} vs market mid {round(mid, 4)} "
                 f"for outcome {args.outcome!r} differs by {round(gap, 4)} "
                 f"(> {EXTREME_DISAGREEMENT}). If this extreme disagreement is your "
                 "researched belief, re-run with --confirm-extreme; if not, you "
                 "probably inverted the outcome side (est_prob must be for the "
                 f"named outcome {args.outcome!r}, not its complement).")

    row = {
        "id": uuid.uuid4().hex[:12],
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "venue" in info:
        row["venue"] = info["venue"]
    row.update({
        "market_id": args.market_id,
        "question": info["question"],
        "slug": info["slug"],
        "end_date": info["end_date"],
        "outcome": args.outcome,
        "token_id": info["token_id"],
        "est_prob": args.est_prob,
        "best_bid_at_record": bid,
        "best_ask_at_record": ask,
        "market_prob_at_record": round(mid, 4),
        "liquidity_at_record": info["liquidity_at_record"],
        "volume_24h_at_record": info["volume_24h_at_record"],
    })
    for key in ("taker_fee_bps_at_record", "topics"):
        if key in info:
            row[key] = info[key]
    row.update({
        "category": args.category,
        "skip_reason": args.skip_reason,
        "fit_score": args.fit_score,
        "note": args.note,
        "strategy_rev": args.strategy_rev,
        "status": "open",
    })
    path = forecasts_path(args.venue)
    if old is not None:
        # The old row stays open so resolve.py still settles it (score.py
        # grades it in the revised-away slice, not the headline stats).
        row["supersedes"] = old["id"]
        old["superseded_by"] = row["id"]
        old["superseded_ts"] = row["ts"]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows + [row]))
    else:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    out = {"recorded": row["id"], "mid": row["market_prob_at_record"],
           "bid": bid, "ask": ask, "delta_vs_mid": round(args.est_prob - mid, 4),
           "question": row["question"]}
    if old is not None:
        out["supersedes"] = old["id"]
        out["delta_est_prob"] = round(args.est_prob - old["est_prob"], 4)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--venue", choices=["polymarket", "myriad"], default="polymarket",
                   help="myriad: --market-id is the Myriad market slug "
                        "(from core/myriad.py scan)")
    p.add_argument("--market-id", required=True)
    p.add_argument("--outcome", required=True, help="exact outcome name, e.g. Yes")
    p.add_argument("--est-prob", type=float, required=True,
                   help="agent's honest probability, formed before reading the price")
    p.add_argument("--category", required=True,
                   help="agent-assigned category, e.g. earnings/soccer/esports/news")
    p.add_argument("--skip-reason", required=True,
                   help="funnel disposition: bet|no-edge|market-agrees|... "
                        "(use 'bet' when a place follows this forecast)")
    p.add_argument("--supersede", action="store_true",
                   help="replace this market+outcome's live open forecast with a "
                        "materially changed read (links the rows; the old one is "
                        "graded in the revised-away slice, not the headline stats)")
    p.add_argument("--confirm-extreme", action="store_true",
                   help="required when |est_prob - mid| > "
                        f"{EXTREME_DISAGREEMENT}: confirms the extreme "
                        "disagreement is deliberate, not an inverted outcome side")
    p.add_argument("--fit-score", type=int, default=None, help="playbook fit score 0-5")
    p.add_argument("--note", default="", help="one line of context (optional)")
    p.add_argument("--strategy-rev", default="", help="git rev of strategy/ used")
    st = sub.add_parser("status")
    st.add_argument("--venue", choices=["polymarket", "myriad"], default="polymarket")
    args = ap.parse_args()

    if args.cmd == "record" and args.venue == "myriad" and args.market_id.startswith("myriad:"):
        args.market_id = args.market_id[len("myriad:"):]
    rows = read_forecasts(forecasts_path(args.venue))
    if args.cmd == "status":
        cmd_status(rows)
    else:
        cmd_record(args, rows)


if __name__ == "__main__":
    main()
