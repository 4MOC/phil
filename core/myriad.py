#!/usr/bin/env python3
"""Myriad Markets API helpers (read-only) and the venue's protected filter.

PROTECTED CORE — the trading agent must not edit files under core/.

Why this venue exists (operator, 2026-10-14): Polymarket answers HTTP 451 to
this fork's machine (South Korea blocked it on 2026-08-18). Myriad Markets'
terms (updated 2026-01-13) do not restrict South Korea and its read endpoints
are public, so it is the phase-A venue: stake-free forecasts only, scored
against Myriad's own order book and settled by Myriad's own resolution.

What is admissible is decided HERE, not by the agent, from the "venues.myriad"
block of config/protected.json:
  * order-book markets only (tradingModel "ob") - an AMM price is not a book
    a paper order could cross, so it is not a benchmark this experiment uses;
  * real-money collateral only (USD1); play-money points (PTS) are excluded;
  * binary markets only, no perpetual markets;
  * no sports or esports markets (topic, sport/league/team tags, moneyline):
    the National Sports Promotion Act bars supplying information for
    unlicensed sports betting, and this repo's journal is public;
  * the protected banned_question_patterns and min_minutes_to_resolution.

Prices: the list and detail records carry outcomes[].bestBid/bestAsk in 0-1
units; GET /markets/<slug>/orderbook?outcome=N returns [price, size] pairs
scaled by 1e18. best_prices() reads the order book - the record values are
used only for the scan's cheap preview.

Resolution: state "resolved" with resolvedOutcomeId naming the winner, or
voided true. Myriad publishes no resolution time (resolvesAt is null), and
expiresAt is not one: "hit in <month>" markets resolve weeks before it, others
after it. So core/resolve.py stamps settled_ts with the UTC hour it first saw
the resolution - never in the future, and the same for any runner settling
the row in that hour.

Usage:
  scan:   python3 core/myriad.py scan [--hours 720] [--limit 400]
          JSON lines, one admissible open market per line.
  market: python3 core/myriad.py market <slug>
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api-v2.myriadprotocol.com"
UA = {"User-Agent": "phil-paper-forecaster/0.1"}
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCALE = 10 ** 18
# Public reads are limited to 30 requests per 10 seconds per IP.
MIN_INTERVAL_S = 0.4
_last_call = [0.0]


class Blocked(RuntimeError):
    """The venue refused this machine (HTTP 403/451); retrying cannot help."""


def protected():
    return json.loads((ROOT / "config" / "protected.json").read_text())


def venue_config(prot=None):
    prot = prot if prot is not None else protected()
    return (prot.get("venues") or {}).get("myriad") or {}


def get_json(path, params=None, retries=3):
    url = API + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (403, 451):
                raise Blocked(f"GET {url} refused: HTTP {e.code} {e.reason}") from e
            if e.code == 404:
                raise RuntimeError(f"GET {url}: HTTP 404 not found") from e
            last_err = e
            retry_after = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit()
                       else 2.0 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 — retry then surface
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last_err}")


def _page_all(params, seen, out, limit):
    """Page one sort order, merging unseen records into out. Returns the
    API's reported total."""
    total, page = None, 1
    while len(out) < limit:
        d = get_json("/markets", dict(params, limit=100, page=page))
        pagination = d.get("pagination") or {}
        total = pagination.get("total", total)
        batch = d.get("data") or []
        for m in batch:
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                out.append(m)
        if not batch or not pagination.get("hasNext"):
            break
        page += 1
    return total


# Paging order matters (review 2026-10-14): expires_at has large ties (about
# 100 markets expire at the same minute), the server orders a tie differently
# at each offset, and about 9% of the open book never appeared on any page.
# published_at paged all 302 of 302 the same day; expires_at stays as a
# second pass that can only add records.
SORT_PASSES = ("published_at", "expires_at")


def markets(state="open", trading_model="ob", limit=400):
    """(records, total): open markets in expiry order, de-duplicated by id,
    and the API's own count. len(records) < min(total, limit) means the API
    did not hand over every market - callers must say so, never imply a
    complete census."""
    out, seen, total = [], set(), None
    for sort in SORT_PASSES:
        total = _page_all({"state": state, "trading_model": trading_model,
                           "sort": sort, "order": "asc"}, seen, out, limit) or total
        if total is not None and len(out) >= min(total, limit):
            break
    out.sort(key=lambda m: m.get("expiresAt") or "")
    return out[:limit], total


def market(slug):
    return get_json(f"/markets/{urllib.parse.quote(slug, safe='')}")


def _levels(side):
    out = []
    for level in side or []:
        try:
            out.append((int(level[0]) / SCALE, int(level[1]) / SCALE))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def book(slug, outcome_index):
    """(bids, asks) for one outcome as lists of (price, shares)."""
    d = get_json(f"/markets/{urllib.parse.quote(slug, safe='')}/orderbook",
                 {"outcome": outcome_index})
    return _levels(d.get("bids")), _levels(d.get("asks"))


def best_prices(slug, outcome_index):
    """Return (best_bid, best_ask) for one outcome, None if that side is empty."""
    bids, asks = book(slug, outcome_index)
    return (max(p for p, _ in bids) if bids else None,
            min(p for p, _ in asks) if asks else None)


def parse_ts(value):
    """Myriad timestamp ('2026-10-14T16:00:00.000Z') -> aware datetime, or None."""
    if not value or not isinstance(value, str):
        return None
    try:
        t = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def iso_z(value):
    t = parse_ts(value)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def outcome_titles(m):
    return [o.get("title") for o in m.get("outcomes") or []]


def token_id(m, outcome_index):
    return f"myriad:{m.get('networkId')}:{m.get('id')}:{outcome_index}"


def is_sports(m, cfg):
    topics = set(m.get("topics") or [])
    if topics & set(cfg.get("excluded_topics") or []):
        return True
    tag_types = {t.get("type") for t in m.get("tags") or [] if isinstance(t, dict)}
    if tag_types & set(cfg.get("excluded_tag_types") or []):
        return True
    return bool(m.get("moneyline"))


def inadmissible(m, now, prot=None):
    """Protected filter. Returns the reason a market may not be used, or None.

    Applies to scan output AND to every forecast record, so a market cannot be
    used by passing its slug directly."""
    prot = prot if prot is not None else protected()
    cfg = venue_config(prot)
    if not cfg:
        return "venues.myriad missing from config/protected.json"
    if m.get("state") != "open":
        return f"state {m.get('state')!r} (not open)"
    if m.get("tradingModel") not in (cfg.get("trading_models") or []):
        return f"trading model {m.get('tradingModel')!r} not allowed"
    symbol = (m.get("token") or {}).get("symbol")
    if symbol not in (cfg.get("collateral_symbols") or []):
        return f"collateral {symbol!r} not allowed"
    if len(m.get("outcomes") or []) != 2:
        return "not a binary market"
    if m.get("perpetual"):
        return "perpetual market"
    if is_sports(m, cfg):
        return "sports/esports market (excluded)"
    title = m.get("title") or ""
    if any(re.search(p, title, re.I) for p in prot.get("banned_question_patterns") or []):
        return "banned question pattern"
    expires = parse_ts(m.get("expiresAt"))
    if expires is None:
        return "no usable expiresAt"
    if expires < now + dt.timedelta(minutes=prot["min_minutes_to_resolution"]):
        return "resolves too soon"
    return None


def resolution(m):
    """('void', None), ('won-by', <winning outcome title>) or (None, None) while
    the market is unresolved."""
    if m.get("state") != "resolved":
        return None, None
    if m.get("voided"):
        return "void", None
    win = m.get("resolvedOutcomeId")
    for o in m.get("outcomes") or []:
        if o.get("id") == win:
            return "won-by", o.get("title")
    return None, None


def scan_record(m):
    outs = m.get("outcomes") or []
    return {
        "venue": "myriad",
        "market_id": m.get("slug"),
        "question": m.get("title"),
        "end_date": iso_z(m.get("expiresAt")),
        "topics": m.get("topics") or [],
        "outcomes": [o.get("title") for o in outs],
        "outcome_prices": [o.get("price") for o in outs],
        "best_bids": [o.get("bestBid") for o in outs],
        "best_asks": [o.get("bestAsk") for o in outs],
        "volume_24h": m.get("volume24h"),
        "liquidity": m.get("liquidity"),
        "taker_fee_bps": (m.get("fees") or {}).get("taker_fee_bps"),
        "resolution_source": m.get("resolutionSource"),
        "description": (m.get("description") or "")[:500],
    }


def cmd_scan(args):
    prot = protected()
    if not venue_config(prot).get("forecasts_enabled"):
        sys.exit("myriad: venues.myriad.forecasts_enabled is false in config/protected.json")
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(hours=args.hours)
    kept, dropped = 0, {}
    try:
        pool, total = markets(limit=args.limit)
    except Blocked as e:
        sys.exit(f"myriad: {e}")
    if total is not None and len(pool) < min(total, args.limit):
        dropped["not paged (API returned fewer markets than its total)"] = (
            min(total, args.limit) - len(pool))
    for m in pool:
        why = inadmissible(m, now, prot)
        if why is None and parse_ts(m.get("expiresAt")) > horizon:
            why = "beyond --hours"
        prices = [p for p in (o.get("price") for o in m.get("outcomes") or [])
                  if isinstance(p, (int, float))]
        if why is None and prices and (max(prices) > prot["max_entry_price"]
                                       or min(prices) < prot["min_entry_price"]):
            why = "effectively decided (price outside protected bounds)"
        # A one-sided book has no mid: its lone quote would stand in for the
        # market's probability. forecast.py rejects it against the live book;
        # this preview check only keeps such markets out of the pool.
        if why is None and any(o.get("bestBid") is None or o.get("bestAsk") is None
                               for o in m.get("outcomes") or []):
            why = "one-sided or empty book"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        kept += 1
        print(json.dumps(scan_record(m)))
    print(f"myriad scan: {kept} admissible within {args.hours}h of {len(pool)} paged "
          f"({total} open order-book markets per the API); "
          f"dropped {dict(sorted(dropped.items()))}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--hours", type=float, default=720)
    s.add_argument("--limit", type=int, default=400, help="max open markets paged")
    mk = sub.add_parser("market")
    mk.add_argument("slug")
    args = ap.parse_args()
    if args.cmd == "scan":
        cmd_scan(args)
    else:
        m = market(args.slug)
        now = dt.datetime.now(dt.timezone.utc)
        rec = scan_record(m)
        rec["inadmissible"] = inadmissible(m, now)
        rec["state"] = m.get("state")
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
