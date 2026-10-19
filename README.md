# Register-anchored study of label leakage in finfluencer classification

Code and released data for the manuscript *Label Leakage in Text Classification of Regulated Status: Evidence from Registered and Unregistered Financial Influencers on YouTube*.

The study links Indian YouTube channels to the public registers of SEBI-registered Investment Advisers and Research Analysts, trains text and behaviour classifiers of registration status, measures how much of the accuracy comes from registration-mandated wording (the *leakage gap*), and validates the resulting ranking of unregistered accounts with two blind human coders.

## Repository layout

```
finfluencer/            analysis code (Python)
  common.py             paths, I/O, content rules, regulatory-wording scrubber
  scrape_sebi_register.py, build_register.py     SEBI registers -> clean register table
  collect_youtube.py    YouTube Data API v3 collection (resumable, quota aware)
  label_accounts.py     register linkage, resolution rules, registration-status labels
  eval_pu.py            positive-unlabelled evaluation (registered vs unregistered)
  nlp_experiments.py    corpus statistics, leakage gap with intervals, scrubbing intensity,
                        nested tuning, representation/classifier ablation, highest-weight terms
  balance_check.py      class-imbalance sensitivity
  topk_validation.py    blind two-coder validation of the ranking (make / kappa / score / bounds)
  eval_flag.py          supervised pilot on the coded behaviour
  active_learning_sim.py  annotation-efficiency simulation
  validate_rules.py     precision of the keyword rules on real hits
  simulation_leakage.py Monte Carlo study of the leakage-gap estimator (bias, bootstrap coverage, incomplete scrubber)
  sample_size.py        sample-size calculations
  scrape_sebi_orders.py, fetch_order_text.py     scoping of SEBI enforcement orders
  export_release.py     builds data/released/ from the private data
  make_paper_figures.py figures for the manuscript
data/released/          public data (hashed identifiers, labels, features, coder judgements, result tables)
docs/                   data dictionary, ethics and data-protection notes
scripts/run_pipeline.sh ordered list of commands
tests/                  unit tests
```

## Requirements

Python 3.11 or later (developed with Python 3.14), and the packages in `requirements.txt` (developed with pandas 3.0, scikit-learn 1.9, NumPy, SciPy, requests, pypdf, matplotlib).

```
pip install -r requirements.txt
python -m unittest discover tests
```

## What is and is not released

Released in `data/released/`: one row per analysed account with a salted-hash identifier, registration status and the rule that produced it, ten numeric platform-behaviour features, the out-of-fold ranking score, the validation group, both coders' judgements, and every results table used in the manuscript.

Not released: channel names and URLs, video titles and descriptions, comments, register names and numbers, and the hashing salt. Raw platform text is excluded to respect the platform's terms of service and the privacy of commenters. The scripts reproduce the raw dataset from the public sources given an API key; because the YouTube content changes over time, a re-collection will not be identical to the dataset analysed in the manuscript.

## Reproducing the analysis from raw data

Set the environment variables, then run the commands in `scripts/run_pipeline.sh`.

```
YT_API_KEY      YouTube Data API v3 key
FIN_SALT        long random string used to hash commenter and account identifiers (keep private, keep fixed)
CONTACT_EMAIL   contact address placed in the user-agent of the SEBI scrapers
FIN_DATA        optional: data directory (default: ./data/finfluencer)
```

The registers are read from the public SEBI listings; the scrapers send one request roughly every 1.5 seconds and stop on any error status. If the site rejects scripted access, the `--from-html` option of `scrape_sebi_register.py` reads pages saved from a browser.

## Correspondence between manuscript and code

| Manuscript item | Script |
|---|---|
| Register, collection and label counts | `scrape_sebi_register.py`, `build_register.py`, `collect_youtube.py`, `label_accounts.py` |
| Corpus statistics | `nlp_experiments.py stats` |
| Registration prediction, baselines | `eval_pu.py` |
| Leakage gap, leakage-channel decomposition | `nlp_experiments.py gap` |
| Scrubbing-intensity experiment | `nlp_experiments.py scrub` |
| Representation, classifier, regularisation and vocabulary ablation | `nlp_experiments.py ablation` |
| Nested tuning of C and vocabulary size (tuned leakage gap) | `nlp_experiments.py tune` |
| Highest-weight terms | `nlp_experiments.py terms` |
| Class-imbalance sensitivity | `balance_check.py` |
| Blind validation, coder agreement | `topk_validation.py` |
| Supervised pilot | `eval_flag.py` |
| Annotation efficiency | `active_learning_sim.py` |
| Keyword-rule validation | `validate_rules.py` |
| Monte Carlo study of the leakage-gap estimator | `simulation_leakage.py` |
| Sample-size calculations | `sample_size.py` |
| Figures 1 to 10 | `make_paper_figures.py` (Figure 9 is drawn by `active_learning_sim.py`) |
| Enforcement-order scoping | `scrape_sebi_orders.py`, `fetch_order_text.py` |

## Responsible use

Unregistered status is not evidence of unlawful activity: educators, news channels and brokers' education arms need no adviser registration. The models in this repository rank accounts by resemblance to registered advisers; the manuscript shows that this ranking does not identify adviser-like behaviour and should not be used to accuse any account. See `docs/ETHICS.md`.

## Licence and citation

Code: MIT licence (see `LICENSE`). Released data: CC BY 4.0. Please cite the manuscript; see `CITATION.cff`.
