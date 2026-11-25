# Limitations

This is personal research, not a production-ready classifier. The answer key is AI-graded. AI labels may share biases with the teacher or reference and are not independent proof of truth. Findings are limited to captured pages and the specified rubric and population. Missing or unassessable evidence is unknown, not negative.

Fresh site-level confidence intervals are wide. The sample was enriched for rare niches and weighted: five sites contributed 45.4% of estimated commercial page-topic positives and five accounted for 65.8% of estimated operator positives. A small number of weighted examples can move subgroup scores substantially. Adult operator detection improved in round two but remained uncertain; French and Polish results were weak or unstable. Some rare topics have sparse support.

The 0.725 A7 result uses the observed graded-page site target. Applying predictions to every captured page against that incomplete target yields 0.570; ungraded true positives may count as false positives. Neither number is a validated population guarantee. The LLM reference did not score every crawl page. The post-hoc combined rule has not been independently confirmed.

The main data gap is real promotional or sponsored placements on publisher websites. Gather more real examples and difficult editorial negatives across niches and languages, with site-disjoint splits and independent human review. Retain page-level evidence and audit model, feature, crawl and threshold versions. Calibrate locally; do not rely on scores for automatic punitive decisions.

The original feature vocabulary and destination lists are not released. They were derived from the crawl and part came from a GPL source. Supplying different lists or projecting the 19 and 16 feature channels differently changes scores and invalidates direct comparison. The numeric weights may encode patterns learned from training data even though raw examples and destination lists are excluded. Training new heads on your own data is strongly recommended.
