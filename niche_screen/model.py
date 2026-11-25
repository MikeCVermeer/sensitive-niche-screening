"""A small, trainable shared-intent baseline, inspired by the research.

No pretrained artifacts are included. A page is a dict with site, embedding,
text, links (already categorized destination labels), and is_homepage. Labels
are dicts keyed by niche, with 1, 0, or None for unassessed.
"""
from collections import Counter, defaultdict
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORD = re.compile(r"\w+", re.UNICODE)
INTENT = {"sponsored", "advertisement", "offer", "buy", "bonus", "signup", "partner", "discount"}


def _words(text):
    return Counter(WORD.findall(text.casefold()))


def _page_features(page, niche, lexicon):
    words = _words(page["text"])
    topic = sum(words[w.casefold()] for w in lexicon[niche])
    intent = sum(words[w] for w in INTENT)
    links = sum(label == niche for label in page.get("links", []))
    embedding = np.asarray(page["embedding"], dtype=float)
    if embedding.ndim != 1 or not np.isfinite(embedding).all():
        raise ValueError("embedding must be a finite one-dimensional vector")
    # Topic and intent interactions allow a single head to transfer the meaning
    # of commercial intent across niches, while keeping topic evidence explicit.
    return np.r_[embedding, np.log1p([topic, intent, links]),
                 np.log1p(topic) * np.log1p(intent), np.log1p(links) * np.log1p(intent)]


def _head():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=7))


def _fit_binary(head, x, y, description):
    if len(set(y)) != 2:
        raise ValueError(f"{description} needs both positive and negative assessed labels")
    head.fit(np.asarray(x), np.asarray(y))


class SharedIntentScreen:
    """Shared page head and separate pooled site-operator head.

    Fit only on training sites and select thresholds on separate development
    sites. The input link labels must be generated independently of test labels.
    """
    def __init__(self, lexicon):
        if not lexicon or any(not terms for terms in lexicon.values()):
            raise ValueError("each niche needs at least one lexical term")
        self.lexicon = lexicon
        self.niches = list(lexicon)
        self.page_head = _head()
        self.operator_head = _head()
        self.dimension = None

    def fit_pages(self, pages):
        if not pages:
            raise ValueError("no pages supplied")
        self.dimension = len(pages[0]["embedding"])
        x, y = [], []
        for page in pages:
            for niche in self.niches:
                label = page["commercial"].get(niche)
                if label is not None:
                    x.append(self._features(page, niche))
                    y.append(int(label))
        _fit_binary(self.page_head, x, y, "page head")
        return self

    def _features(self, page, niche):
        if len(page["embedding"]) != self.dimension:
            raise ValueError("inconsistent embedding dimension")
        return _page_features(page, niche, self.lexicon)

    def page_scores(self, pages):
        return [{n: float(self.page_head.predict_proba([self._features(p, n)])[0, 1])
                 for n in self.niches} for p in pages]

    def site_features(self, pages):
        if not pages:
            raise ValueError("a site needs at least one captured page")
        if len({page['site'] for page in pages}) != 1:
            raise ValueError("site features must contain exactly one site")
        scores = self.page_scores(pages)
        features = {}
        for niche in self.niches:
            values = np.array([s[niche] for s in scores])
            home = [s[niche] for p, s in zip(pages, scores) if p.get("is_homepage", False)]
            topic_counts = [sum(_words(p["text"])[w.casefold()] for w in self.lexicon[niche]) for p in pages]
            links = sum(sum(link == niche for link in p.get("links", [])) for p in pages)
            features[niche] = [values.max(), values.mean(), np.mean(values >= .5),
                               max(home, default=0), bool(home), np.log1p(sum(topic_counts)), np.log1p(links)]
        return features

    def fit_operators(self, pages, site_labels):
        groups = group_pages(pages)
        x, y = [], []
        for site, group in groups.items():
            features = self.site_features(group)
            for niche in self.niches:
                label = site_labels.get(site, {}).get(niche)
                if label is not None:
                    x.append(features[niche])
                    y.append(int(label))
        _fit_binary(self.operator_head, x, y, "operator head")
        return self

    def predict_sites(self, pages, page_threshold=.5, operator_threshold=.5):
        """Return distinct component scores and decisions; no implicit OR rule."""
        out = {}
        for site, group in group_pages(pages).items():
            scores = self.page_scores(group)
            features = self.site_features(group)
            out[site] = {}
            for niche in self.niches:
                highest = max(row[niche] for row in scores)
                operator = float(self.operator_head.predict_proba([features[niche]])[0, 1])
                out[site][niche] = {"page_score": highest, "commercial": highest >= page_threshold,
                                    "operator_score": operator, "operator": operator >= operator_threshold}
        return out


def group_pages(pages):
    groups = defaultdict(list)
    for page in pages:
        groups[page["site"]].append(page)
    return dict(groups)
