"""Dense retriever over data/corpus.jsonl.

Query  = problem description + ground-truth spec (never a solution).
Index  = each entry's `query` text, embedded once and cached to data/emb_*.npy.
Score  = cosine similarity; lemmas and examples are ranked separately.

Falls back to a TF-IDF retriever if sentence-transformers is not installed
(say which one you used in the report).
"""
import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

import config


class Retriever:
    def __init__(self, corpus_path=config.DATA_DIR / "corpus.jsonl", model=config.EMBED_MODEL):
        self.corpus = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
        texts = [c["query"] for c in self.corpus]
        try:
            if not model:
                raise ImportError("EMBED_MODEL is None -> TF-IDF")
            from sentence_transformers import SentenceTransformer
            self.backend = "dense:" + model
            self.model = SentenceTransformer(model)
            key = hashlib.sha1((model + "".join(texts)).encode()).hexdigest()[:12]
            cache = config.DATA_DIR / f"emb_{key}.npy"
            if cache.exists():
                self.emb = np.load(cache)
            else:
                self.emb = self.model.encode(texts, batch_size=128, show_progress_bar=True,
                                             normalize_embeddings=True).astype(np.float32)
                np.save(cache, self.emb)
        except Exception as e:      # package missing, or model download blocked/offline
            print(f"warning: dense retriever unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  "using TF-IDF. Report this in the paper if it happens in a real run.")
            self.backend = "tfidf"
            self.model = None
            self._tfidf_fit(texts)

    # ---------------------------------------------------------------- TF-IDF fallback
    @staticmethod
    def _tokens(text):
        return re.findall(r"[A-Za-z]+|\d+", text.replace("_", " ").replace(".", " ").lower())

    def _tfidf_fit(self, texts):
        docs = [Counter(self._tokens(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vocab = {w: i for i, w in enumerate(self.idf)}
        self.emb = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for w, c in d.items():
                self.emb[i, self.vocab[w]] = (1 + math.log(c)) * self.idf[w]
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9

    def _encode(self, text):
        if self.model is not None:
            return self.model.encode([text], normalize_embeddings=True)[0]
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for w, c in Counter(self._tokens(text)).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[w]
        return v / (np.linalg.norm(v) + 1e-9)

    # ---------------------------------------------------------------- query
    BOILERPLATE = re.compile(r"--[^\n]*|\b(def|problem_spec|implementation|impl|spec|result|let|"
                             r"if|then|else|Prop|Bool)\b|∃ result, impl[^\n]*")
    TYPES = ("List", "Nat", "Int", "String", "Rat", "Char", "Finset", "Option", "Array")

    @classmethod
    def spec_symbols(cls, problem):
        """Types and functions the spec + signature actually use, e.g.
        ({'List','Int'}, {'take','sum','length'})."""
        text = problem["problem_spec_formal_ground_truth"] + "\n" + problem["implementation_signature"]
        types = {t for t in cls.TYPES if re.search(rf"\b{t}\b", text)}
        funcs = set(re.findall(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", text))
        funcs |= {f.split(".")[-1] for f in re.findall(r"\b(?:%s)\.[a-zA-Z_]\w*" % "|".join(cls.TYPES), text)}
        if "∣" in text:
            funcs.add("dvd")
        return types, funcs

    def lemma_query_text(self, problem):
        spec = self.BOILERPLATE.sub(" ", problem["problem_spec_formal_ground_truth"])
        types, funcs = self.spec_symbols(problem)
        return " ".join(sorted(types | funcs)) + "\n" + spec

    def name_bonus(self, name, types, funcs):
        """+0.15 if the lemma is about a used type AND mentions a used function."""
        parts = name.split(".")
        tokens = set(re.split(r"_", parts[-1]))
        if parts[0] in types and tokens & funcs:
            return 0.15
        if tokens & funcs:
            return 0.05
        return 0.0

    def query(self, problem, k, kind):
        """Top-k entries of one kind ('lemma' or 'example') for a problem view.
        Examples: matched on description + spec. Lemmas: matched on the spec's
        content (boilerplate removed) plus a bonus for names using its symbols."""
        if kind == "lemma":
            q = self._encode(self.lemma_query_text(problem))
            types, funcs = self.spec_symbols(problem)
        else:
            q = self._encode(f"{problem['problem_spec_nl']}\n{problem['problem_spec_formal_ground_truth']}")
        scores = self.emb @ q
        if kind == "lemma":
            scores = scores + np.array([self.name_bonus(c["id"], types, funcs) if c["kind"] == "lemma" else 0
                                        for c in self.corpus], dtype=np.float32)
        out = []
        for i in np.argsort(-scores):
            c = self.corpus[i]
            if c["kind"] != kind or problem["pid"] in c.get("blocked_for", []):
                continue
            out.append(c | {"score": float(scores[i])})
            if len(out) == k:
                break
        return out


if __name__ == "__main__":       # quick look: python retriever.py 2
    import sys
    import data
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    r, p = Retriever(), data.prompt_view(pid)
    print("backend:", r.backend)
    print(p["problem_spec_nl"][:300], "\n")
    for kind, k in (("example", config.N_FEWSHOT), ("lemma", config.N_LEMMAS)):
        for c in r.query(p, k, kind):
            print(f"{c['score']:.3f}  {c['id']:<35} {c['text'][:90]!r}" if kind == "lemma"
                  else f"{c['score']:.3f}  {c['id']}")
