#!/bin/bash
# Ordered commands to rebuild the analysis from public sources. Requires YT_API_KEY, FIN_SALT and CONTACT_EMAIL.
# The YouTube Data API has a default quota of 10,000 units per day; search.list costs 100 units, so the discovery step
# and the comment step are resumable and are meant to be re-run on later days.
set -euo pipefail
cd "$(dirname "$0")/../finfluencer"
: "${YT_API_KEY:?set YT_API_KEY}" "${FIN_SALT:?set FIN_SALT}" "${CONTACT_EMAIL:?set CONTACT_EMAIL}"

# 1. SEBI registers (Investment Advisers: intmId 13, Research Analysts: intmId 14)
python scrape_sebi_register.py --intm-id 13 --name investment_advisers
python scrape_sebi_register.py --intm-id 14 --name research_analysts
python build_register.py --snapshot-date "$(date +%F)"

# 2. YouTube collection (resumable)
python collect_youtube.py --max-units 9000 discover --queries-file queries.txt --pages 2
python collect_youtube.py --max-units 200 channels
python collect_youtube.py --max-units 4000 videos --per-channel 30 --min-subs 1000
python collect_youtube.py --max-units 4000 comments --videos-per-channel 8 --per-video 40

# 3. Labels and analysis pool (see docs/METHODS notes in the manuscript)
python label_accounts.py

# 4. Experiments reported in the manuscript
python sample_size.py
python eval_pu.py --repeats 3 --folds 5
python nlp_experiments.py stats
python nlp_experiments.py terms
python nlp_experiments.py scrub
python nlp_experiments.py ablation
python nlp_experiments.py gap
python nlp_experiments.py tune
python simulation_leakage.py --reps 200
python leakage_miner.py overlap
python leakage_miner.py real
python leakage_miner.py sim --reps 200
python embedding_baseline.py embed
python embedding_baseline.py cv
python robustness_checks.py groups
python robustness_checks.py deletion
python balance_check.py --repeats 3

# 5. Human validation: create the blind sheets, have two coders fill them independently, then score
python topk_validation.py make --n-top 100 --n-control 50
#   ... coding by two people (see docs/CODING_GUIDE.md) ...
python topk_validation.py kappa
python topk_validation.py bounds
python eval_flag.py --repeats 10 --folds 5
python active_learning_sim.py --repeats 100

# 6. Public release tables and figures
python export_release.py --out ../data/released
python make_paper_figures.py
