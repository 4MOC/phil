#!/usr/bin/env python3
"""Offline tests for the Myriad venue (core/myriad.py) and its wiring into
core/forecast.py, core/resolve.py and core/pmapi.py's 451 handling.

PROTECTED CORE - the trading agent must not edit files under core/.

Everything here runs on hand-built records shaped like Myriad's API answers
(observed 2026-10-14); the network is never touched. CI runs it on every push.

Usage: python3 core/test_myriad.py
"""
import argparse
import copy
import datetime as dt
import email.message
import io
import json
import pathlib
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import forecast  # noqa: E402
import myriad  # noqa: E402
import pmapi  # noqa: E402
import resolve  # noqa: E402
import score  # noqa: E402

NOW = dt.datetime(2026, 10, 14, 3, 0, tzinfo=dt.timezone.utc)
SLUG = "btc-price-on-october-18-4pm-utc-btc-above-84000-on-october-18"
E18 = 10 ** 18


def mk(**over):
    m = {
        "id": 3700, "networkId": 56, "slug": SLUG,
        "title": "BTC above $84,000 on October 18, 4PM UTC?",
        "description": "Resolves Yes if the Binance BTC/USDT 1-minute candle ...",
        "expiresAt": "2026-10-18T16:00:00.000Z", "resolvesAt": None,
        "fees": {"maker_fee_bps": 0, "taker_fee_bps": 98},
        "state": "open", "voided": False, "resolvedOutcomeId": -1,
        "topics": ["Crypto"],
        "tags": [{"slug": "crypto", "type": "category", "title": "Crypto"}],
        "resolutionSource": "https://www.binance.com/en/trade/BTC_USDT",
        "token": {"symbol": "USD1", "decimals": 18},
        "liquidity": 1349.5, "volume24h": 12.0,
        "perpetual": False, "moneyline": False, "inPlay": False,
        "executionMode": 1, "tradingModel": "ob",
        "outcomes": [
            {"id": 0, "title": "Yes", "price": 0.46, "bestBid": 0.45, "bestAsk": 0.48},
            {"id": 1, "title": "No", "price": 0.54, "bestBid": 0.52, "bestAsk": 0.55},
        ],
    }
    m.update(over)
    return m


def book_json(bids, asks):
    return {"bids": [[str(int(p * E18)), str(int(s * E18))] for p, s in bids],
            "asks": [[str(int(p * E18)), str(int(s * E18))] for p, s in asks]}


# A fixture config: the tests must not turn red when the operator later
# switches Myriad forecasts off or edits the live patterns.
PROT = copy.deepcopy(myriad.protected())
PROT.setdefault("venues", {})["myriad"] = {
    "forecasts_enabled": True, "bets_enabled": False, "sim_bankroll_usd": 1000.0,
    "trading_models": ["ob"], "collateral_symbols": ["USD1"],
    "excluded_topics": ["Sports", "Gaming"],
    "excluded_tag_types": ["sport", "league", "team"]}
PROT["banned_question_patterns"] = ["Up or Down - .*[0-9]:[0-9]{2}[AP]M-[0-9]"]
PROT["min_minutes_to_resolution"] = 20


class Admissible(unittest.TestCase):
    def test_live_config_never_enables_myriad_bets(self):
        # The one live invariant validate.py also enforces (phase A).
        self.assertIsNot(myriad.venue_config().get("bets_enabled"), True)

    def test_order_book_crypto_market_is_admissible(self):
        self.assertIsNone(myriad.inadmissible(mk(), NOW, PROT))

    def test_rejections(self):
        cases = {
            "sports topic": mk(topics=["Sports"]),
            "esports topic": mk(topics=["Gaming"]),
            "team tag": mk(topics=["Culture"], tags=[{"type": "team", "title": "Dodgers"}]),
            "moneyline": mk(topics=["Culture"], moneyline=True),
            "amm": mk(tradingModel="amm", executionMode=0),
            "play money": mk(token={"symbol": "PTS"}),
            "multi-outcome": mk(outcomes=mk()["outcomes"] + [{"id": 2, "title": "Draw"}]),
            "perpetual": mk(perpetual=True),
            "resolved": mk(state="resolved"),
            "too soon": mk(expiresAt="2026-10-14T03:05:00.000Z"),
            "no expiry": mk(expiresAt=None),
            "banned": mk(title="Bitcoin Up or Down - October 14, 9:00AM-9:15AM ET"),
        }
        for name, m in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(myriad.inadmissible(m, NOW, PROT))

    def test_missing_venue_block_admits_nothing(self):
        prot = dict(PROT)
        prot.pop("venues")
        self.assertIsNotNone(myriad.inadmissible(mk(), NOW, prot))


class Prices(unittest.TestCase):
    def test_best_prices_unscale_the_book(self):
        d = book_json([(0.45, 13.1), (0.44, 50)], [(0.48, 127), (0.50, 10)])
        with mock.patch.object(myriad, "get_json", return_value=d) as g:
            self.assertEqual(myriad.best_prices(SLUG, 0), (0.45, 0.48))
        self.assertEqual(g.call_args.args[1], {"outcome": 0})

    def test_empty_side_is_none(self):
        with mock.patch.object(myriad, "get_json", return_value=book_json([(0.98, 5)], [])):
            self.assertEqual(myriad.best_prices(SLUG, 0), (0.98, None))


def fake_pages(pages_by_sort, total):
    """get_json stub serving /markets pages per sort key; pages overlap the
    way Myriad's tied expires_at pages do."""
    def get_json(path, params=None, retries=3):
        pages = pages_by_sort.get(params["sort"], [])
        i = params["page"] - 1
        ids = pages[i] if i < len(pages) else []
        return {"data": [mk(id=n, slug=f"m{n}") for n in ids],
                "pagination": {"total": total, "hasNext": i + 1 < len(pages)}}
    return mock.patch.object(myriad, "get_json", side_effect=get_json)


class Paging(unittest.TestCase):
    def test_second_sort_pass_recovers_markets_hidden_by_ties(self):
        with fake_pages({"published_at": [[1, 2, 3], [3, 4, 5]],
                         "expires_at": [[6, 1, 2], [3, 4, 5]]}, total=6):
            got, total = myriad.markets()
        self.assertEqual((sorted(m["id"] for m in got), total), ([1, 2, 3, 4, 5, 6], 6))

    def test_complete_first_pass_skips_the_second(self):
        with fake_pages({"published_at": [[1, 2, 3], [4, 5, 6]]}, total=6) as g:
            got, _ = myriad.markets()
        self.assertEqual(len(got), 6)
        self.assertTrue(all(c.args[1]["sort"] == "published_at" for c in g.call_args_list))

    def test_shortfall_is_visible_to_the_caller(self):
        with fake_pages({"published_at": [[1, 2, 3], [3, 4, 5]],
                         "expires_at": [[1, 2, 3], [3, 4, 5]]}, total=6):
            got, total = myriad.markets()
        self.assertLess(len(got), total)


class Resolution(unittest.TestCase):
    def test_states(self):
        self.assertEqual(myriad.resolution(mk()), (None, None))
        self.assertEqual(myriad.resolution(mk(state="closed")), (None, None))
        self.assertEqual(myriad.resolution(mk(state="resolved", resolvedOutcomeId=1)),
                         ("won-by", "No"))
        self.assertEqual(myriad.resolution(mk(state="resolved", voided=True,
                                              resolvedOutcomeId=-1)), ("void", None))

    def test_settle_row_stamps_the_hour_it_was_seen(self):
        # A "hit in October" market resolves weeks before its expiresAt; the
        # stamp is the observing hour, never the future expiry.
        row = {"outcome": "Yes", "status": "open", "end_date": "2026-11-01T03:59:00Z"}
        now = dt.datetime(2026, 10, 14, 3, 47, 12, tzinfo=dt.timezone.utc)
        self.assertFalse(resolve.settle_against_myriad(dict(row), mk(), now))
        won = dict(row)
        self.assertTrue(resolve.settle_against_myriad(
            won, mk(state="resolved", resolvedOutcomeId=0,
                    expiresAt="2026-11-01T03:59:00.000Z"), now))
        self.assertEqual((won["status"], won["outcome_won"], won["settled_ts"]),
                         ("won", "Yes", "2026-10-14T03:00:00Z"))
        later = now + dt.timedelta(minutes=10)
        again = dict(row)
        resolve.settle_against_myriad(again, mk(state="resolved", resolvedOutcomeId=0), later)
        self.assertEqual(again["settled_ts"], won["settled_ts"])
        lost = dict(row)
        resolve.settle_against_myriad(lost, mk(state="resolved", resolvedOutcomeId=1), now)
        self.assertEqual(lost["status"], "lost")
        void = dict(row)
        resolve.settle_against_myriad(void, mk(state="resolved", voided=True), now)
        self.assertEqual(void["status"], "void")


class Record(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        self.poly, self.myr = d / "forecasts.jsonl", d / "forecasts-myriad.jsonl"
        for name, value in (("FORECASTS", self.poly), ("FORECASTS_MYRIAD", self.myr)):
            p = mock.patch.object(forecast, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(myriad, "protected", return_value=PROT)
        p.start()
        self.addCleanup(p.stop)
        self.book = book_json([(0.45, 13)], [(0.48, 127)])

    def args(self, **over):
        a = dict(venue="myriad", market_id=SLUG, outcome="Yes", est_prob=0.55,
                 category="crypto", skip_reason="no-edge", supersede=False,
                 confirm_extreme=False, fit_score=None, note="", strategy_rev="abc1234")
        a.update(over)
        return argparse.Namespace(**a)

    def fake_api(self, market):
        def get_json(path, params=None, retries=3):
            if path.endswith("/orderbook"):
                return self.book
            return market
        return mock.patch.object(myriad, "get_json", side_effect=get_json)

    # cmd_record checks admissibility against the real clock, so the fixture
    # market must expire in the future whenever CI runs.
    EXPIRES = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=4)).replace(
        microsecond=0)

    def record(self, market, **over):
        market = dict(market, expiresAt=self.EXPIRES.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        with self.fake_api(market), mock.patch("sys.stdout", io.StringIO()):
            forecast.cmd_record(self.args(**over), forecast.read_forecasts(self.myr))

    def test_writes_only_the_myriad_file(self):
        self.record(mk())
        self.assertFalse(self.poly.exists())
        rows = forecast.read_forecasts(self.myr)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["venue"], "myriad")
        self.assertEqual(r["market_id"], SLUG)
        self.assertEqual(r["token_id"], "myriad:56:3700:0")
        self.assertEqual((r["best_bid_at_record"], r["best_ask_at_record"]), (0.45, 0.48))
        self.assertEqual(r["market_prob_at_record"], 0.465)
        self.assertEqual(r["end_date"], self.EXPIRES.strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertEqual(r["taker_fee_bps_at_record"], 98)
        self.assertEqual(r["status"], "open")

    def test_sports_market_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self.record(mk(topics=["Sports"]))
        self.assertIn("not admissible", str(cm.exception))
        self.assertFalse(self.myr.exists())

    def test_unknown_outcome_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.record(mk(), outcome="Maybe")

    def test_one_sided_book_is_rejected(self):
        self.book = book_json([(0.98, 5)], [])
        with self.assertRaises(SystemExit) as cm:
            self.record(mk())
        self.assertIn("one-sided", str(cm.exception))
        self.assertFalse(self.myr.exists())

    def test_disabled_venue_is_rejected(self):
        off = copy.deepcopy(PROT)
        off["venues"]["myriad"]["forecasts_enabled"] = False
        with mock.patch.object(myriad, "protected", return_value=off), \
                self.assertRaises(SystemExit):
            self.record(mk())

    def test_one_live_row_per_market_outcome(self):
        self.record(mk())
        with self.assertRaises(SystemExit):
            self.record(mk(), est_prob=0.56)
        self.record(mk(), est_prob=0.62, supersede=True)
        rows = forecast.read_forecasts(self.myr)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["superseded_by"], rows[1]["id"])


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "Unavailable For Legal Reasons",
                                  email.message.Message(), None)


class Blocked(unittest.TestCase):
    def test_pmapi_451_raises_at_once(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(451)), \
                mock.patch("time.sleep") as sleep:
            with self.assertRaises(pmapi.Blocked):
                pmapi.gamma_market("123")
        sleep.assert_not_called()

    def test_myriad_403_raises_at_once(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(403)), \
                mock.patch("time.sleep"):
            with self.assertRaises(myriad.Blocked):
                myriad.market(SLUG)

    def test_resolve_asks_a_blocked_venue_once(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        rows = [{"id": str(i), "market_id": str(100 + i), "outcome": "Yes",
                 "status": "open", "end_date": "2026-10-01T00:00:00Z"} for i in range(3)]
        (d / "forecasts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with mock.patch.object(resolve, "LEDGER", d / "ledger.jsonl"), \
                mock.patch.object(resolve, "FORECASTS", d / "forecasts.jsonl"), \
                mock.patch.object(resolve, "FORECASTS_MYRIAD", d / "forecasts-myriad.jsonl"), \
                mock.patch.object(resolve.pmapi, "gamma_market",
                                  side_effect=pmapi.Blocked("HTTP 451")) as gm, \
                mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            resolve.main()
        self.assertEqual(gm.call_count, 1)
        after = [json.loads(x) for x in (d / "forecasts.jsonl").read_text().splitlines()]
        self.assertEqual([r["status"] for r in after], ["open"] * 3)

    def test_mark_to_market_asks_a_blocked_venue_once(self):
        positions = [{"id": str(i), "token_id": str(i)} for i in range(4)]
        with mock.patch.object(score.pmapi, "best_prices",
                               side_effect=pmapi.Blocked("HTTP 451")) as bp:
            rows = score.mark_to_market(positions)
        self.assertEqual(bp.call_count, 1)
        self.assertEqual([r["id"] for r in rows], ["0", "1", "2", "3"])
        self.assertTrue(all("error" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
