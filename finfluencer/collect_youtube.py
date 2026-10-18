"""Stage 1 - collect public YouTube data (YouTube Data API v3). Resumable, quota-aware.

    set YT_API_KEY=...   set FIN_SALT=...
    python collect_youtube.py discover --queries-file queries.txt --max-units 9000
    python collect_youtube.py channels
    python collect_youtube.py videos --per-channel 30
    python collect_youtube.py comments --videos-per-channel 10 --per-video 100

Default daily quota is 10,000 units: search.list = 100, every other call used here = 1.
So `discover` is the bottleneck (~90 queries/day); everything else is cheap. Re-run on later
days - finished queries/ids are skipped. Request a quota extension for the study if needed.
Only public data are collected. Commenter ids are salted-hashed; names are never stored.
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import time

import requests

from common import RAW, append_jsonl, pseudonym, read_jsonl

API = "https://www.googleapis.com/youtube/v3/"
COST = {"search": 100}


class QuotaExceeded(Exception):
    pass


class Client:
    def __init__(self, max_units):
        self.key = os.environ.get("YT_API_KEY") or sys.exit("Set YT_API_KEY")
        self.units, self.max_units = 0, max_units

    def get(self, endpoint, **params):
        cost = COST.get(endpoint, 1)
        if self.units + cost > self.max_units:
            raise QuotaExceeded(f"local budget of {self.max_units} units reached")
        params["key"] = self.key
        for attempt in range(5):
            try:
                r = requests.get(API + endpoint, params=params, timeout=30)
            except requests.RequestException as e:  # transient SSL/connection drops; never print the URL (it holds the key)
                print(f"  network error ({type(e).__name__}), retry {attempt + 1}/5")
                time.sleep(3 * 2 ** attempt)
                continue
            if r.status_code == 200:
                self.units += cost
                return r.json()
            try:
                err = r.json().get("error", {})
            except ValueError:
                err = {}
            reasons = {e.get("reason") for e in err.get("errors", [])}
            if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                raise QuotaExceeded("YouTube daily quota exhausted")
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return {"_error": r.status_code, "_reasons": sorted(x for x in reasons if x)}
        return {"_error": "retries_exhausted"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def batches(xs, n=50):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def discover(c, a):
    done_f = RAW / "discover_done.json"
    done = set(json.loads(done_f.read_text(encoding="utf-8"))) if done_f.exists() else set()
    if a.seeds:  # channel ids found by other means, e.g. register entries searched by name
        ids = [l.strip() for l in open(a.seeds, encoding="utf-8") if l.strip().startswith("UC")]
        append_jsonl(RAW / "discovery.jsonl", [{"channel_id": i, "query": "SEED", "rank": 0, "retrieved": now()} for i in ids])
    queries = [q.strip() for q in open(a.queries_file, encoding="utf-8") if q.strip() and not q.startswith("#")]
    try:
        for q in queries:
            if q in done:
                continue
            token, rank = None, 0
            for _ in range(a.pages):
                p = dict(part="snippet", q=q, type="channel", maxResults=50, regionCode="IN")
                if token:
                    p["pageToken"] = token
                js = c.get("search", **p)
                if "_error" in js:
                    print("search error", js)
                    break
                rows = []
                for it in js.get("items", []):
                    rank += 1
                    rows.append({"channel_id": it["snippet"]["channelId"], "query": q, "rank": rank, "retrieved": now()})
                append_jsonl(RAW / "discovery.jsonl", rows)
                token = js.get("nextPageToken")
                if not token:
                    break
            done.add(q)
            done_f.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")
            print(f"[discover] {q!r}: units used {c.units}")
    except QuotaExceeded as e:
        print("stopped:", e)


def channels(c, a):
    have = {r["channel_id"] for r in read_jsonl(RAW / "channels.jsonl")}
    want = {r["channel_id"] for r in read_jsonl(RAW / "discovery.jsonl")} - have
    try:
        for chunk in batches(sorted(want)):
            js = c.get("channels", part="snippet,statistics,contentDetails", id=",".join(chunk), maxResults=50)
            rows = []
            for it in js.get("items", []):
                s, st = it.get("snippet", {}), it.get("statistics", {})
                rows.append({"channel_id": it["id"], "title": s.get("title"), "description": s.get("description"),
                             "custom_url": s.get("customUrl"), "country": s.get("country"),
                             "published_at": s.get("publishedAt"),
                             "subscribers": int(st["subscriberCount"]) if "subscriberCount" in st else None,
                             "subs_hidden": st.get("hiddenSubscriberCount", False),
                             "views": int(st.get("viewCount", 0)), "video_count": int(st.get("videoCount", 0)),
                             "uploads_playlist": it.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads"),
                             "retrieved": now()})
            append_jsonl(RAW / "channels.jsonl", rows)
            print(f"[channels] +{len(rows)}")
    except QuotaExceeded as e:
        print("stopped:", e)


def only_ids(a):
    return {l.strip() for l in open(a.ids_file, encoding="utf-8") if l.strip()} if getattr(a, "ids_file", None) else None


def videos(c, a):
    done = {r["channel_id"] for r in read_jsonl(RAW / "videos.jsonl")} | \
           {r["channel_id"] for r in read_jsonl(RAW / "videos_empty.jsonl")}
    chans = [r for r in read_jsonl(RAW / "channels.jsonl") if r["channel_id"] not in done and r.get("uploads_playlist")]
    chans = [r for r in chans if (r["subscribers"] or 0) >= a.min_subs]
    keep = only_ids(a)
    if keep is not None:
        chans = [r for r in chans if r["channel_id"] in keep]
    try:
        for ch in chans:
            ids, token = [], None
            while len(ids) < a.per_channel:
                p = dict(part="contentDetails", playlistId=ch["uploads_playlist"], maxResults=50)
                if token:
                    p["pageToken"] = token
                js = c.get("playlistItems", **p)
                if "_error" in js:
                    break
                ids += [i["contentDetails"]["videoId"] for i in js.get("items", [])]
                token = js.get("nextPageToken")
                if not token:
                    break
            ids = ids[:a.per_channel]
            rows = []
            for chunk in batches(ids):
                js = c.get("videos", part="snippet,statistics,contentDetails", id=",".join(chunk))
                for it in js.get("items", []):
                    s, st = it["snippet"], it.get("statistics", {})
                    rows.append({"channel_id": ch["channel_id"], "video_id": it["id"], "title": s.get("title"),
                                 "description": s.get("description"), "tags": s.get("tags", []),
                                 "published_at": s.get("publishedAt"),
                                 "language": s.get("defaultAudioLanguage") or s.get("defaultLanguage"),
                                 "duration": it.get("contentDetails", {}).get("duration"),
                                 "views": int(st.get("viewCount", 0)),
                                 "likes": int(st["likeCount"]) if "likeCount" in st else None,
                                 "comments": int(st["commentCount"]) if "commentCount" in st else None,
                                 "retrieved": now()})
            if rows:
                append_jsonl(RAW / "videos.jsonl", rows)
            else:
                append_jsonl(RAW / "videos_empty.jsonl", [{"channel_id": ch["channel_id"]}])
            print(f"[videos] {ch['title']!r}: {len(rows)} (units {c.units})")
    except QuotaExceeded as e:
        print("stopped:", e)


def comments(c, a):
    done = {r["video_id"] for r in read_jsonl(RAW / "comments_done.jsonl")}
    by_ch = {}
    for v in sorted(read_jsonl(RAW / "videos.jsonl"), key=lambda r: r["published_at"] or "", reverse=True):
        by_ch.setdefault(v["channel_id"], []).append(v)
    keep = only_ids(a)
    order = sorted(ch for ch in by_ch if keep is None or ch in keep)
    random.Random(42).shuffle(order)  # random channel order: a quota-truncated run must not be biased toward discovery order
    try:
        for ch in order:
            vs = by_ch[ch]
            for v in vs[:a.videos_per_channel]:
                if v["video_id"] in done or v.get("comments") == 0:
                    continue
                js = c.get("commentThreads", part="snippet", videoId=v["video_id"], order="relevance",
                           maxResults=min(a.per_video, 100), textFormat="plainText")
                rows = []
                for it in js.get("items", []):
                    s = it["snippet"]["topLevelComment"]["snippet"]
                    author = s.get("authorChannelId", {}).get("value")
                    rows.append({"channel_id": ch, "video_id": v["video_id"], "comment_id": it["id"],
                                 "author": pseudonym(author) if author else None, "text": s.get("textOriginal"),
                                 "likes": s.get("likeCount"), "published_at": s.get("publishedAt"),
                                 "by_channel_owner": author == ch})
                append_jsonl(RAW / "comments.jsonl", rows)
                append_jsonl(RAW / "comments_done.jsonl",
                             [{"video_id": v["video_id"], "status": js.get("_reasons") or "ok"}])
            print(f"[comments] {ch}: units {c.units}")
    except QuotaExceeded as e:
        print("stopped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-units", type=int, default=9000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--queries-file", default="queries.txt")
    d.add_argument("--pages", type=int, default=2)
    d.add_argument("--seeds")
    sub.add_parser("channels")
    v = sub.add_parser("videos")
    v.add_argument("--per-channel", type=int, default=30)
    v.add_argument("--min-subs", type=int, default=1000)
    v.add_argument("--ids-file", help="only these channel ids (one per line)")
    m = sub.add_parser("comments")
    m.add_argument("--videos-per-channel", type=int, default=8)   # sized in sample_size.py
    m.add_argument("--per-video", type=int, default=40)
    m.add_argument("--ids-file", help="only these channel ids (one per line)")
    a = ap.parse_args()
    {"discover": discover, "channels": channels, "videos": videos, "comments": comments}[a.cmd](Client(a.max_units), a)
