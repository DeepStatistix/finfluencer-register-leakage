# Coding guide: do these YouTube accounts operate like investment advisers?

Fill `coder_A.csv` (or `coder_B.csv`) independently. Do not discuss with the other coder until both are done.
Each row is one channel. Everything you need is in the row (channel description + 6 recent video titles and description starts).
You do NOT need to watch videos. If a row is truly unclear, open the channel link and read the About page or ONE recent video
description (not the video). About 2 minutes per row. You cannot see any model score or group, on purpose.

The accounts are NOT registered with SEBI as Investment Advisers or Research Analysts. Being unregistered is not illegal
for educators or news. The question is only what the account does.

## Columns to fill

### gives_specific_recommendations  (1 or 0)
1 = the channel regularly names specific securities / indices / contracts AND tells viewers what to do or expect:
    buy/sell/hold, entry, target, stop-loss, "stock to buy", "multibagger pick", "BankNifty tomorrow, buy above X".
0 = general education ("what is a stop loss"), news, company or economy analysis with no call, market commentary,
    broker/app how-tos, or crypto/forex education without calls.

### sells_tips_or_advisory  (1 or 0)
1 = the channel sells or promotes access to tips, signals, calls, advisory, portfolios, "premium"/VIP groups, or paid
    trading courses that include calls.
0 = free-only content, ads for a platform/app, or courses that are pure education with no calls or signals.
    "Join our free Telegram" alone is 0.

### entity_type  (choose ONE word exactly)
- educator                  teaches markets/personal finance; individual or academy
- news_media                news or media company, journalist, business channel
- other_regulated_entity    stock broker, bank, mutual fund house, AMC, insurer, exchange, depository (regulated by SEBI/RBI/IRDAI in another capacity)
- individual_adviser_like   an individual/firm that acts like an adviser or analyst (research, views, calls, advisory)
- crypto_forex_trader       crypto, forex, binary options or commodities trading content
- other                     anything else

## Rules of thumb
- Judge the account, not one video. Look at several rows of titles/descriptions.
- A disclaimer ("not investment advice") does not change what the account does. Judge the behaviour.
- If an account says it IS a SEBI registered adviser, still code what you see (the register says it is not on it).
- Unsure between 1 and 0: pick 0, and write the doubt in `notes`.
- Do not google the channel owner beyond the About page.
- Save as CSV (keep the file name and all column names).

When both coders are done: `python topk_validation.py kappa` (agreement + the list to adjudicate), then
`python topk_validation.py score`.
