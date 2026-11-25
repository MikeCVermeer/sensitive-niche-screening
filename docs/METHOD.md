# Case study and method

## Problem and approach

A problem encountered at Seeders prompted my own research on a crawl of hundreds of thousands of websites and numerous pages per site(millions in total). The challenge is not only recognizing a topic. A news report about a topic, an affiliate offer and an operator's own sales page have different relationships to it. A single site may have an ordinary homepage and a promotional page elsewhere.

I tested nine niches: gambling, consumer loans, adult, dating, CBD, crypto, e-cigarettes, forex and pharmacy. The commercial-page screen estimates relationships for each topic and aggregates the strongest page at site level. A separate pooled head estimates whether a site is an operator. These outputs are review signals, not catalogue decisions. Capture coverage and missing vectors are not negative labels.

The chosen A7 head combines a 1,024-dimensional existing page vector with lexical and outbound-link features, prototype similarities, and interactions. Its page head is shared across niches, so evidence about commercial intent can transfer to sparse topics. A development-set calibration converts raw relation scores to alarm scores. The separately trained round-two site head pools topic, offer and homepage features, plus semantics from a weighted site vector. Its threshold was fixed before the fresh evaluation.

## Two rounds

Round one compared seven low-cost alternatives, a page-level LLM and an independent multilingual encoder. Site-disjoint development and test cohorts, blind grading, inverse-inclusion weighting and site-cluster bootstrap intervals were used. A4 was the original development choice; A7 had the best frozen test site-alarm result, so it was carried forward as a candidate, not presented as a development-selected winner.

Round two expanded topics and languages, trained a site operator head, compared architecture and data ablations, and froze a development selection before a new 615-site cohort. The fresh results favored keeping A7 for page screening and using the pooled operator score separately. A post-hoc OR of the two added only two site-topic flags across that cohort and needs independent confirmation.

## How I used AI in this project

An LLM teacher generated training labels; AI also helped write synthetic edge cases, grade a blind answer key, and assist exploration and implementation. An RTX 4090 was used for the independent multilingual encoder comparison.

## Costs and boundaries

The two rounds used approximately $7 in metered OpenAI calls, excluding compute, crawling and other costs. A per-page LLM pass was estimated at hundreds of dollars per full run at the measured rate and crawl scale, not a provider quote or an invoice. With compatible embeddings already available, the lightweight heads add no per-page API charge; creating embeddings, crawling and running infrastructure still costs money. Measurements and assumptions are in [RESULTS](RESULTS.md).
