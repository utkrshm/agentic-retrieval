"""Dummy end-to-end demo: rank a small built-in snippet library for NL queries on CPU.

    uv run python scripts/demo.py "How is the input preprocessed before going to the main function?"
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # query time is CPU-only

from coderet.config import MODELS, get_model  # noqa: E402
from coderet.models.encoders import SentenceTransformerEmbedder  # noqa: E402
from coderet.mteb_adapters import PrePostPipelineEncoder  # noqa: E402
from coderet.retrieval.dense import Retriever  # noqa: E402

SNIPPETS = {
    # the three snippets from the theme-1 guidelines
    "guide/normalize.js": "function normalize(str) {\n  const str2 = str.trim();\n  return forward(str2);\n}",
    "guide/check.js": "function check(s) {\n  var pre = s.slice(0,6);\n  return pre === 'en-US';\n}",
    "guide/perf.js": "function perf(str) {\n  if (act(A, str)) {\n    return act(B, str)\n  }\n}",
    # a toy library
    "util/retry.py": "def retry(fn, attempts=3, delay=0.5):\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            time.sleep(delay * 2 ** i)\n    raise RuntimeError('gave up')",
    "util/slugify.py": "def slugify(title):\n    s = re.sub(r'[^a-z0-9]+', '-', title.lower())\n    return s.strip('-')",
    "util/chunks.py": "def chunks(xs, n):\n    for i in range(0, len(xs), n):\n        yield xs[i:i + n]",
    "util/lru.py": "class LRU:\n    def __init__(self, cap):\n        self.cap, self.d = cap, OrderedDict()\n    def get(self, k):\n        self.d.move_to_end(k)\n        return self.d[k]\n    def put(self, k, v):\n        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.cap:\n            self.d.popitem(last=False)",
    "util/debounce.js": "function debounce(fn, ms) {\n  let t;\n  return (...args) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), ms);\n  };\n}",
    "auth/hash.py": "def hash_password(pw, salt):\n    return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, 200_000).hex()",
    "auth/jwt.js": "function verifyToken(token, secret) {\n  const [h, p, sig] = token.split('.');\n  if (sign(h + '.' + p, secret) !== sig) throw new Error('bad signature');\n  return JSON.parse(atob(p));\n}",
    "io/read_csv.py": "def read_rows(path):\n    with open(path, newline='') as f:\n        return [row for row in csv.DictReader(f)]",
    "io/config.py": "def load_config(path):\n    with open(path) as f:\n        cfg = yaml.safe_load(f)\n    return {**DEFAULTS, **cfg}",
    "algo/binary_search.py": "def lower_bound(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo",
    "algo/dijkstra.py": "def dijkstra(graph, src):\n    dist = {src: 0}\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        for v, w in graph[u]:\n            if d + w < dist.get(v, inf):\n                dist[v] = d + w\n                heapq.heappush(pq, (d + w, v))\n    return dist",
    "algo/primes.py": "def sieve(n):\n    is_p = [True] * (n + 1)\n    is_p[0:2] = [False, False]\n    for i in range(2, int(n ** .5) + 1):\n        if is_p[i]:\n            is_p[i*i::i] = [False] * len(is_p[i*i::i])\n    return [i for i, p in enumerate(is_p) if p]",
    "algo/lis.py": "def lis(a):\n    tails = []\n    for x in a:\n        i = bisect_left(tails, x)\n        tails[i:i+1] = [x]\n    return len(tails)",
    "web/route.py": "@app.get('/users/{uid}')\ndef get_user(uid: int):\n    user = db.get(uid)\n    if user is None:\n        raise HTTPException(404)\n    return user",
    "web/fetch.js": "async function getJson(url) {\n  const res = await fetch(url);\n  if (!res.ok) throw new Error(res.status);\n  return res.json();\n}",
    "voice/intent.js": "function routeIntent(req) {\n  switch (req.intent.name) {\n    case 'OpenBluetoothSettings': return deeplink('settings://bluetooth');\n    case 'SetAlarm': return alarms.create(req.slots.time);\n    default: return fallback(req);\n  }\n}",
    "voice/wakeword.py": "def detect_wakeword(frames, model, threshold=0.8):\n    for frame in frames:\n        if model.score(mfcc(frame)) > threshold:\n            return True\n    return False",
    "text/tokenize.py": "def tokenize(text):\n    text = unicodedata.normalize('NFKC', text).lower()\n    return re.findall(r'\\w+', text)",
}

QUERIES = [
    "How is the input preprocessed before going to the main function?",
    "retry a failing call with exponential backoff",
    "which code opens the bluetooth settings page?",
    "find shortest paths from a source node in a weighted graph",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="*", default=QUERIES)
    parser.add_argument("--model", default="jina-code-0.5b", choices=sorted(MODELS))
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    print(f"model={args.model}  device=cpu  snippets={len(SNIPPETS)}")
    retriever = Retriever.build(PrePostPipelineEncoder(SentenceTransformerEmbedder(get_model(args.model), device="cpu")), SNIPPETS)
    retriever.search("warm up")
    for query in args.queries:
        hits = retriever.search(query, args.k)
        t = retriever.last_timings_ms
        print(f"\nQ: {query}   [encode {t['encode']:.0f} ms | scan {t['scan']:.2f} ms | total {t['total']:.0f} ms]")
        for rank, (doc, score) in enumerate(hits, 1):
            print(f"  {rank}. {score:.3f}  {doc:22s} {SNIPPETS[doc].splitlines()[0]}")


if __name__ == "__main__":
    main()
