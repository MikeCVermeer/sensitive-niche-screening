# Shared intent screening

A small, trainable toolkit for investigating whether websites carry commercial content in sensitive topics. It was inspired by research built at Seeders. It uses embeddings already available to a crawler, topic words and categorized outbound destinations; a shared page classifier learns commercial intent across topics. A separate pooled site classifier estimates whether a site operates in a topic. Both return scores and separate decisions.

The included implementation is an **educational A7-style reimplementation**, not the original model or a reproduction of its reported scores. It has no pretrained weights or real examples. You must bring lawful, labeled, site-disjoint data, compatible embeddings and your own language-specific vocabularies and destination categorization. The toy vectors and text are generated from scratch for the demo. Its apparent accuracy is meaningless.

## Run the demo

Python 3.10 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m niche_screen.demo
python -m unittest discover -s tests -v
```

The demo holds out whole synthetic sites and reports site-level commercial F1 and a site-cluster bootstrap interval. It does not train or evaluate a production classifier. See `niche_screen/demo.py` for the input contract and a minimal example. A page contains a site identifier, embedding, text, optional categorized links, homepage flag, and per-topic commercial labels. Use `None` for unassessed labels; do not turn missing evidence into a negative. Site operator labels are supplied separately. All embeddings must come from the same model and have the same dimensionality.

`SharedIntentScreen.fit_pages` trains the shared page head; `fit_operators` trains a separate site head from pooled page signals. `predict_sites` returns each head's score and decision. Its default 0.5 thresholds are illustrative: select thresholds using a separate development set. Group splits by site, check duplicates across splits, and use out-of-fold page scores when training the site head to reduce stacking leakage. Never train or tune on your held-out test set. Capture coverage and unassessable sites need explicit handling by the caller.

`site_bootstrap` resamples sites, retaining all their topic rows, and reports micro F1 and a percentile interval. It assumes equally sampled sites and is not a substitute for design-weighted evaluation. If your sample is stratified, enriched or has unequal inclusion probabilities, implement an appropriate weighted estimator. The original research used sampling weights; this simple harness cannot reproduce those estimates.

The prompts in `prompts/` are starting points for a teacher and synthetic training examples. Review their output, protect private source material, and keep evaluation independent. The [method and lessons](METHOD.md) explain research results and their limitations.

MIT licensed; see `LICENSE`.
