# Method and lessons

## The question

A catalogue of roughly 145,000 supplier sites had about 30 captured pages per site. A topic mention alone is not the same as promotion: an independent report about crypto prices should not automatically be treated as a paid crypto placement. The goal was to find sensitive-topic operators and promotional content without an LLM or expensive API call for every page at inference time.

## What worked

The original A7 candidate shared commercial-intent information across topic-specific page examples, using existing text-embedding-3-small page vectors, lexical evidence and categorized outbound targets. Separately, a pooled site model used homepage and site-wide evidence to distinguish operators from publishers with placements. A maximum over captured page decisions supplied site-level commercial alarms. Reusing existing vectors avoids new embedding or LLM API charges for this step, but crawling, embedding, compute and integration still cost money.

On a fresh cohort of 615 sites across nine languages, the frozen A7 site alarm reached F1 0.725 (95% site-cluster bootstrap interval 0.599–0.830) against the observed graded-page target. An LLM reference reached 0.806 and a fine-tuned multilingual transformer reached 0.802 on the same target. The latter needed another GPU inference pass. The independent pooled operator head scored 0.721, or 0.921 after excluding low-confidence positive operator labels. These are separate results, not the score of the combined system. Intervals are wide and some categories and languages are particularly uncertain.

Scoring every captured page against the same *incomplete* observed truth gave A7 0.570: real positive pages outside the graded sample may be counted as false positives. The exploratory OR of A7 and the newer operator head changed almost nothing and has not been independently confirmed. An older homepage-only LLM workflow scored 0.545 on a different first-round cohort and any-finding target, so that number is context, not a directly paired improvement.

## What did not win

Hand- or LLM-written weights, a self-growing link graph and additional Common Crawl operator pages did not beat the selected approach in the relevant comparisons. A GPU fine-tune approached the LLM reference but required a separate inference pass, a poor trade-off for this deployment. LLM-generated edge cases helped a development comparison, but synthetic-distribution results are not proof of real-world benefit. More real sponsored placements on publisher sites remain the main data gap.

The fresh answer key was graded blindly by an AI model, not confirmed by human review. A human spot check is still pending; production integration has not been built. Evaluate on a new, independently reviewed and representative sample before making decisions. Record capture coverage and retain evidence for review.

## How the research was run

The developer worked with Claude Opus as lead, using structured questions to settle choices and delegating exploration and documentation to GPT-6.1 Terra and implementation and training to GPT-6.2 Astra. Agents were explicitly encouraged to try unusual ideas, including directly writing model weights. A separate gpt-6-luna teacher generated training labels; Claude Opus graded blinded material for the answer key. Two rounds separated development choices from a fresh holdout. A home RTX 4090 supplied the transformer ceiling comparison. Total OpenAI API spend was approximately $7 for the research; this excludes compute and other costs.

The code here is a clean, smaller reimplementation of the idea, with deliberately simple features and an unweighted evaluation harness. It does not implement the original prototype projections, sampling design, calibration, data pipeline, or frozen research models, and cannot reproduce the research numbers. Treat the results above as observations from the research, not claims about this demo.
