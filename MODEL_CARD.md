---
license: mit
language:
- en
- nl
- de
- es
- pt
- sv
- fr
- it
- pl
tags:
- text-classification
- content-moderation
- embeddings
- research
metrics:
- f1
---

# Sensitive-niche screening research weights

Personal research shared for further study, not a production-ready system. The answer key was AI-graded and requires human spot checks. The released numeric A7 page and round-two pooled operator heads need **both** 1,024-dimensional OpenAI `text-embedding-3-small` page vectors and separately projected lexical/link evidence. The original cue and link lists are not included: they were derived from the crawl and part came from a GPL source. This is a feature-level release, not a text-to-score model. [Bring your own data](docs/BRING_YOUR_OWN.md) explains the inputs and retraining.

## Intended use

Exploratory, human-reviewed screening for promotional or operator content in nine sensitive niches. It is not designed for automated penalties, safety certification or inference about people. Missing captures must remain unassessed. Report feature and embedding versions and retain evidence for review.

## Training data and model

The research used page vectors, lexical/link evidence and LLM-generated training labels from a multilingual crawl, plus some synthetic edge cases. Site-level training used pooled evidence. Raw text, examples, labels, identifiers, destination lists and vectors are not released. The NPZ contains numeric coefficients and synthetic prototype directions only; configuration contains thresholds and calibration. Score inputs and language choices may differ materially from the research pipeline.

## Evaluation

On a fresh 615-site cohort, weighted A7 site alarm F1 was 0.725 (95% site-bootstrap interval 0.599–0.830) on graded-page evidence. The LLM reference was 0.806 on that target. A7 all-crawl predictions against incomplete observed truth scored 0.570; ungraded true positives can appear false. The separate site operator head scored 0.721 (0.384–0.921). These are different targets. Sampling was enriched and weighted, with large uncertainty. See [results](docs/RESULTS.md).

## Limitations and bias

The answer key was AI-graded, not human-validated. Performance varies across language and niche and rare groups have sparse support; adult, French and Polish cases warrant particular review. The principal data gap is real sponsored content on publisher sites. Training data and topic vocabulary can underrepresent particular languages, cultures or business models. Different user-supplied features, embedding preparation and thresholds alter behavior. Validate against independently reviewed, representative sites before any practical use. See [limitations](docs/LIMITATIONS.md).
