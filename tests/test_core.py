"""Unit tests for the core logic. They use small hand-written inputs only; none of the data in the study is needed.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finfluencer"))


class Scrubber(unittest.TestCase):
    def test_removes_registration_and_disclosure_sentences(self):
        from common import scrub_regulatory
        text = ("Good level analysis for tomorrow. We are SEBI registered research analysts INH000014340. "
                "The securities quoted are for illustration only. Subscribe to the channel.")
        self.assertEqual(scrub_regulatory(text), "Good level analysis for tomorrow. Subscribe to the channel.")

    def test_word_boundaries(self):
        from common import scrub_regulatory
        self.assertIn("material", scrub_regulatory("Study material for beginners."))       # 'ria' inside 'material'
        self.assertEqual(scrub_regulatory("Registration granted by SEBI does not assure returns."), "")

    def test_partition(self):
        from common import extract_regulatory, scrub_regulatory
        text = "Nifty view. SEBI registered adviser. Join us. Past results are not a recommendation."
        kept, removed = scrub_regulatory(text).split(), extract_regulatory(text).split()
        self.assertEqual(sorted(kept + removed), sorted(text.split()))


class RulesAndRegisterNumbers(unittest.TestCase):
    def test_rules(self):
        from common import registration_numbers, rule_flags
        f = rule_flags("Buy TCS target 4000 SL 3800. 100% sure profit. Join VIP group for fees Rs 5000")
        self.assertTrue(f["trade_call"] and f["guarantee"] and f["paid_solicitation"])
        self.assertFalse(rule_flags("Past performance does not guarantee future returns")["guarantee"])
        self.assertFalse(rule_flags("I am not SEBI registered")["claims_registration"])
        self.assertEqual(registration_numbers("INH000012345, ina000099999 and INAIFSC10001"),
                         ["INA000099999", "INAIFSC10001", "INH000012345"])


class RegistrationResolution(unittest.TestCase):
    ch = {"title": "Example Channel", "description": ""}

    def test_precedence(self):
        from label_accounts import auto_resolve
        denies = {"title": "Example", "description": "I am NOT a SEBI registered adviser. INH000012345"}
        self.assertEqual(auto_resolve(denies, ["INH000012345"], ["Example Research"], 0.0)[0], "unregistered")
        self.assertEqual(auto_resolve(self.ch, [], [], 0.99)[1], "exact_register_name")
        quotes = {"title": "Example", "description": "SEBI RA INH000012345"}
        self.assertEqual(auto_resolve(quotes, ["INH000012345"], ["Someone Else"], 0.2), ("registered", "channel_quotes_register_number"))
        named = {"title": "Rahul Pahuja trading", "description": ""}
        self.assertEqual(auto_resolve(named, ["INH000029467"], ["Rahul Pahuja Proprietor of RAYA CAPITAL"], 0.2)[1], "holder_name_in_channel_text")
        self.assertEqual(auto_resolve(self.ch, ["INH000000001"], ["Unrelated Person"], 0.1)[0], "ambiguous")
        self.assertEqual(auto_resolve(self.ch, [], [], 0.9)[0], "unregistered")


class RegisterParser(unittest.TestCase):
    def test_parse_both_quote_styles(self):
        from scrape_sebi_register import parse
        html = ("<div class=\"title\"><span>Name</span></div><div class=\"value x\"><span>A &amp; Co</span></div>"
                "<div class='title'><span>Registration No.</span></div><div class='value'><span>INA000017523</span></div>"
                "<div class=\"title\"><span>E-mail</span></div><div class=\"value\"><span>x&#64;y.in</span></div>"
                "<div class='title'><span>Validity</span></div><div class='value'><span>Dec 22, 2022 - Perpetual</span></div>")
        rows = parse(html)
        self.assertEqual(rows, [{"Name": "A & Co", "Registration No.": "INA000017523", "Validity": "Dec 22, 2022 - Perpetual"}])  # e-mail dropped


class Statistics(unittest.TestCase):
    def test_sample_size_helpers(self):
        from sample_size import n_kappa, n_proportion
        self.assertEqual(n_proportion(0.5, 0.05), 385)
        self.assertEqual(n_proportion(0.8, 0.07), 126)
        self.assertEqual(n_kappa(0.85, 0.5, 0.12), 137)

    def test_wilson_interval(self):
        from topk_validation import wilson
        lo, hi = wilson(48, 100)
        self.assertAlmostEqual(lo, 0.385, places=2)
        self.assertAlmostEqual(hi, 0.577, places=2)


if __name__ == "__main__":
    unittest.main()
