# Experiments and ablations

## Round one: frozen test

The weighted page alarm F1 and max-over-graded-pages site alarm F1 below use the same round-one blind test and nine topics; see [results](RESULTS.md) for cohort and uncertainty. The reference LLM incurs a per-page cost. mmBERT is evaluated in a separate ceiling run.

| Approach | Page alarm F1 | Site alarm F1 | What happened |
|---|---:|---:|---|
| A1 hand/LLM-written rules and weights | 0.497 | 0.565 | Explicit weighted signals missed too much context. |
| A2 multilingual prototypes | 0.347 | 0.338 | Similarity alone did not adequately separate topic from promotion. |
| A3 page-vector probe | 0.534 | 0.574 | A learned embedding head improved on prototypes. |
| A4 page probe plus features | 0.659 | 0.683 | Lexical and link evidence improved the probe; development-selected. |
| A5 self-growing destination graph | 0.497 | 0.565 | No improvement on the frozen test. |
| A6 max chunk probes | 0.525 | 0.486 | Chunk scoring added work without a site-level gain. |
| A7 shared intent probe | 0.736 | 0.780 | Shared intent and topic features led the low-cost test. |
| LLM on every sampled page | 0.855 | 0.889 | Highest reference, with a marginal API cost. |
| Fine-tuned mmBERT | 0.763 | 0.767 | Competitive but requires a new encoder pass. |

The older homepage-only baseline scored 0.545 on a different eight-topic, 148-site paired any-finding target; it cannot be compared directly to the table. The GPU reference took 1.26 seconds per 1,000 pages on the measured RTX 4090, excluding data preparation. Hardware, batch size and integration change this result.

## Round two: development comparisons

| Variant | Page commercial F1 | Site operator F1 | Observed site alarm F1 |
|---|---:|---:|---:|
| Frozen R1 A7, mapped | 0.769 | 0.267 | 0.745 |
| Shared head, no external examples | 0.702 | 0.857 | 0.690 |
| Vector-only head | 0.381 | 0.857 | 0.667 |
| Expanded features | 0.735 | 0.857 | 0.755 |
| Legacy scores | 0.733 | 0.857 | 0.826 |
| Expanded plus graph | 0.723 | 0.857 | 0.741 |
| Selected shared head, without Common Crawl operators | 0.731 | 0.857 | 0.851 |

On development data, pooled operator features scored 0.857, domain vector 0.667 and their ensemble 0.600. Max page aggregation scored 0.851 versus 0.606 for a two-page count and 0.647 for the top-three mean. Selection was frozen before fresh evaluation, where the selected round-two page head underperformed A7; the operator head remained useful as a distinct signal.

## External-data ablations

With the expanded architecture fixed, original-only training scored 0.735 page commercial and 0.755 site alarm on development. Adding all eligible external sources (928 rows) gave 0.753 and 0.784; synthetic examples alone (461 rows) gave the same point estimates. Adding only Common Crawl operator pages (466 rows) did not change either point estimate. Only one eligible production-linking row was available, so it supplies no meaningful standalone conclusion. Synthetic evaluation is not proof of transfer to real placements. A separate public-source sensitivity experiment affected development scores inconsistently and was not tested on the fresh holdout. The full aggregate JSON and original reports contain more comparisons; no raw source material is included here.
