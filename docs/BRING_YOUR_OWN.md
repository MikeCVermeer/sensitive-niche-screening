# Bring your own data

This repository shares research weights and a feature-level API. It does not include the original cue vocabulary, crawl-matched destinations or graph: they were derived from the crawl and part came from a GPL source. You must build your own evidence projector and validate or retrain on your data. The listed channel meanings below are a contract, not a claim that substituting your own vocabulary reproduces the research features or quality. Do not interpret a missing list, failed capture or missing embedding as a negative.

## 1. Own your cues and destinations

An illustrative, wholly fictional file is [own-lists.json](../examples/own-lists.json). `cues` maps topic names to terms; `destinations` maps exact lowercase hostnames to topic names. Use your own lawful and reviewed terms, with no URLs or paths as keys. The helper does exact host matching, not suffix or subdomain matching, and does no network access:

```python
from niche_screen.inputs import load_lists, categorize_links
cues, destinations = load_lists('examples/own-lists.json')
labels = categorize_links(['https://example.test/page'], destinations)
```

You may collect your own labels or investigate public sources such as [Wikidata](https://www.wikidata.org/) for structured entities and [The Block List Project](https://github.com/blocklistproject/Lists) for example domain lists. Neither is a ready-made answer key for this task. Check each source's current license, file-level terms, attribution and redistribution restrictions, and evaluate whether its inclusion rules match your intended use. The research's matched list is excluded even when a source is publicly accessible. Beware false matches and treat brands and domains as time-sensitive.

The simple `SharedIntentScreen` baseline in `niche_screen/model.py` accepts a caller-owned `cues` dictionary and page `links` already mapped to topics by the helper. It is a different, smaller architecture and cannot reproduce the research scores. The frozen research heads instead consume preprojected numerical features, described below; the helper does not produce those features for you.

## 2. Create compatible embeddings

Use OpenAI `text-embedding-3-small` with `dimensions=1024`. The research used unprefixed, title/heading-aware page chunks, normalized each chunk, averaged them, and normalized the mean. A single request for a whole short page is a useful starting point but is not guaranteed to match that chunking pipeline. Record your text preparation, model version and chunking; API calls incur cost and send text to the provider under its terms. Example for one already-prepared chunk (requires the separately installed `openai` client and your own authentication):

```python
from openai import OpenAI
import numpy as np

client = OpenAI()
result = client.embeddings.create(model='text-embedding-3-small', dimensions=1024,
                                  input=['A fictional page about games.'])
v = np.asarray(result.data[0].embedding, dtype=np.float32)
v /= np.linalg.norm(v)
```

For multiple chunks normalize each, average, and normalize again. Never create zero vectors for failed captures; the released API rejects them. This snippet does not implement the original chunker.

## 3. Project your evidence

All arrays follow topic order `gambling, loan, adult, dating, cbd, crypto, esmoke, forex, pharmacy`. Each site call contains pages from one site. Scores depend on these features, not just the vectors.

`a7_features` has shape `(pages, 9, 19)`. In order, its channels are: topic in title; topic in meta; count of topic-containing prose sentences capped at six; topic in headings; topic in URL path; descriptive topic anchor in main content; generic topic anchor; matched categorized destination; offer associated with topic or a descriptive categorized link; topic anchor with sponsored or affiliate marker; descriptive topic anchor bound to a categorized destination; offer near topic in prose; own-service wording near topic or a titled topic; offer anywhere in body; disclosure; homepage or very short/listing page; `log1p(word_count)/8`; internal topic-anchor count capped at three; graph evidence. The frozen graph was empty; the release includes no graph. The original boundary rules, feature definitions and data are not supplied: the names alone are insufficient to guarantee equivalence.

`operator_features` has shape `(pages, 9, 16)`. Channels 0–3 indicate the original topic vocabulary in title, meta, headings and text excerpt. Channels 4–9 are expanded topic terms in title/meta/headings, expanded body count (capped at five), topic-specific negative cue, CTA in summary, CTA in body, and CTA near an expanded topic term. Channels 10–15 are topic anchor, topic anchor in main content, categorized destination, topic anchor bound to a categorized or named destination in main content, topic anchor with offer, and topic anchor with sponsored rel. This is a separate projection, with a separately chosen excerpt. Categories and lexical rules are yours to define and audit.

Use `load_model('weights')`, then `score_pages(model, vectors, a7_features)` or `score_site(model, vectors, a7_features, operator_features, homepage)`. The package validates array shape and finite values, not whether your projector is semantically compatible. Different features or embeddings change the frozen output. Prefer retraining rather than assuming the research thresholds transfer.

## 4. Train and calibrate on your own sites

Create a numeric `.npz` file with `numpy.savez_compressed` and these arrays. Rows with the same site must stay in one split. Do not include object arrays or raw private text in a shared artifact.

| Array | Shape | Meaning |
|---|---|---|
| `vectors` | `(N,1024)` | normalized nonzero page embeddings |
| `page_features` | `(N,9,19)` | your A7 feature projection |
| `operator_features` | `(N,9,16)` | your site-head page evidence projection |
| `homepage` | `(N,)` | boolean flag per page |
| `site_index` | `(N,)` | contiguous integer site index, 0 through S−1 |
| `site_split` | `(S,)` | 0 train, 1 development, 2 held-out test |
| `page_labels` | `(N,9)` | −1 unassessed, 0 absent, 1 editorial, 2 promotional, 3 operator |
| `operator_labels` | `(S,9)` | −1 unassessed, 0 not operator, 1 operator |

Include positive and negative examples of each of the three page relations in training, positive and negative development examples for alarm, any and operator calibration, and enough sites for a meaningful held-out test. Group duplicates and near-duplicates across sites when assigning splits. Labels and cue definitions should match your rubric. The script uses the released semantic directions, refits both heads on train sites, selects unweighted calibration and thresholds on development sites, then reports an unweighted site-cluster bootstrap on test sites. It is a clear starting recipe, not a reproduction of the original sampling weights and model-selection procedure. The saved NPZ and JSON are loadable by the same inference API.

```sh
python -m niche_screen.retrain --input my-projected-data.npz --output my-model
python -m niche_screen.retrain --demo  # entirely fictional generated data
```

Inspect `niche_screen/retrain.py::synthetic` for a runnable array construction example. Use `niche_screen.evaluation.site_bootstrap` to evaluate further site-topic decisions; it resamples whole sites and ignores unassessed labels. It is unweighted and cannot reproduce estimates for enriched or unequal-probability samples. Use a design-aware evaluator where needed, maintain a final untouched site-held-out test set, and obtain independent human review.
