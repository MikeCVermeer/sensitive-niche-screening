"""Generate an entirely fictional dataset and run train, score and evaluate."""
import json
import numpy as np
from .evaluation import site_bootstrap
from .model import SharedIntentScreen

LEXICON = {"gaming": ["game", "wager"], "finance": ["loan", "credit"]}


def synthetic_data():
    """Invent all text, labels and vectors; no scraped material is used."""
    rng = np.random.default_rng(12)
    pages, operators = [], {}
    niches = list(LEXICON)
    for i in range(40):
        site = f"site-{i:02d}"
        operator_niche = niches[i % 2] if i % 4 == 0 else None
        operators[site] = {n: int(n == operator_niche) for n in niches}
        for j in range(3):
            chosen = niches[(i + j) % 2]
            commercial = (j == 1 and i % 3 == 0) or chosen == operator_niche
            mention = commercial or (i + j) % 3 == 0
            text = (LEXICON[chosen][0] if mention else "garden weather") + (" sponsored offer bonus" if commercial else " independent report")
            vector = rng.normal(0, .12, 6)
            vector[niches.index(chosen)] += 1 if mention else 0
            vector[2] += 1 if commercial else 0
            pages.append({"site": site, "embedding": vector.tolist(), "text": text,
                          "links": [chosen] if commercial else [], "is_homepage": j == 0,
                          "commercial": {n: int(commercial and n == chosen) for n in niches}})
    return pages, operators


def main():
    pages, operators = synthetic_data()
    train = [p for p in pages if int(p["site"][-2:]) < 30]
    test = [p for p in pages if int(p["site"][-2:]) >= 30]
    screen = SharedIntentScreen(LEXICON).fit_pages(train).fit_operators(train, operators)
    predictions = screen.predict_sites(test)
    rows = []
    for site, result in predictions.items():
        for niche, decision in result.items():
            truth = any(p["commercial"][niche] for p in test if p["site"] == site)
            rows.append({"site": site, "label": int(truth), "prediction": int(decision["commercial"])})
    print(json.dumps({"data": "fictional demonstration only", "commercial_site_alarm": site_bootstrap(rows, 500),
                      "example": next(iter(predictions.values()))}, indent=2))


if __name__ == "__main__":
    main()
