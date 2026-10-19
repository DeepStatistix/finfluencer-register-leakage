# Data dictionary

## data/released/accounts.csv

One row per YouTube channel with at least 1,000 subscribers and at least one collected video (795 rows). Identifiers are salted SHA-256 hashes truncated to 12 hexadecimal characters; the salt is private.

| Column | Type | Description |
|---|---|---|
| account_id | string | Salted hash of the YouTube channel identifier. |
| reg_status | categorical | `registered`, `unregistered` or `ambiguous`, derived from the SEBI registers (Section 3.3 of the manuscript). |
| reg_resolution_rule | categorical | Rule that resolved the 43 unclear cases (`channel_quotes_register_number`, `holder_name_in_channel_text`, `exact_register_name`, `channel_denies_registration`, `fuzzy_name_only`, `number_only_in_videos_and_no_name_link`); empty for accounts that matched directly. |
| n_videos | integer | Number of collected videos (at most 30). |
| in_analysis_pool | boolean | True for the 591 accounts used in the models (all registered accounts and a random sample of 450 unregistered accounts). |
| subscribers_decile | integer 1–10 | Decile of subscriber count among the 795 accounts (the exact count is not released). |
| log_subs, log_views_per_sub, log_video_count | numeric | Natural-log transformed audience size features. |
| channel_age_days | numeric | Days between channel creation and the collection date. |
| uploads_per_month | numeric | Videos divided by the months spanned by the collected videos (minimum span 1 month). |
| like_rate, comment_rate | numeric | Sum of likes (comments) divided by sum of views over the collected videos. |
| median_log_views | numeric | Median of log(1 + views) over collected videos. |
| mean_title_len | numeric | Mean video-title length in characters. |
| share_caps_titles | numeric | Share of titles in which more than half of the characters are upper case. |
| pu_out_of_fold_score | numeric | Out-of-fold score from the best registration model (gradient boosting on behaviour and scrubbed text); defined for pool accounts. |
| validation_group | categorical | `top` (100 highest-scoring unregistered accounts) or `control` (50 drawn from the lower-scoring half); empty for accounts outside the validation sample. |
| coderA_*, coderB_* | categorical | The two coders' judgements: `gives_specific_recommendations` (0/1), `sells_tips_or_advisory` (0/1), `entity_type` (educator, news_media, other_regulated_entity, individual_adviser_like, crypto_forex_trader, other). |
| flag_strict, flag_lenient | 0/1 | Adviser-like flag under the strict (both coders) and lenient (either coder) rule, excluding accounts judged to be other regulated entities. |

## data/released/results/

| File | Content |
|---|---|
| pu_results.csv | AUC, average precision and recall at 20% flag rate for registered versus unregistered, by feature set and model. |
| oof_scores.csv | Repeat-averaged out-of-fold score of every account for each feature set and classifier (one column per combination, named features, a vertical bar, then model, for example `text_clean\|logreg`); used for the ROC and precision-recall curves. |
| nlp_oof_summary.csv | Out-of-fold AUC with bootstrap intervals and Brier scores for scrubbed, unscrubbed and removed-wording text and for behaviour. |
| leakage_gap_ci.csv | Leakage gap and scrubbed-text minus behaviour AUC difference with paired bootstrap intervals. |
| scrub_levels.csv | AUC at four scrubbing intensities and the share of characters kept. |
| nlp_ablation.csv | Representation, classifier, regularisation and vocabulary-size ablation. |
| balance_check.csv | Class-imbalance sensitivity. |
| flag_pilot_results.csv | Supervised pilot on the coded behaviour. |
| active_learning_summary.csv, active_learning_curves.csv | Annotation-efficiency simulation (summary and per-repeat curves). |
| sim_leakage_results.csv, sim_leakage_replicates.csv | Monte Carlo study of the leakage-gap estimator: scenario summaries and the estimate and interval of every replicate. |
| corpus_stats.csv | Text-length and script statistics by class. |
| top_terms_scrubbed.csv | Highest-weight terms of the scrubbed logistic regression. |
