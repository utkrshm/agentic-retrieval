# Decisions, advice on record and incidents

## Incidents worth remembering

- **Silent CPU fallback during indexing:** a first re-index hit a CUDA out-of-memory error on its first batch;
  the resilient encoder demoted itself to the CPU for good and the build crawled. Fixed with a GPU batch of 8
  and `expandable_segments`. Two GPU jobs must not run at once on this 4 GB card.
- **Empty cache falsy:** an empty `VectorCache` was falsy through `__len__`, so it never filled until the check
  became `is not None` (found by a test).
- **Council reviews:** rounds on the plan, on quality boosts, on improvement ideas and on a LightGBM reranker.
  Findings used: cut symbol boosts, call maps, agent loop and evidence bundles; add the gated lexical lane; no
  reranker yet (see section 3).

## Advice on record

- **LightGBM reranker:** all three reviewers (Sol, Astra, Opus) said not now. The runtime cost is small (about 2 to 10 ms,
  an estimate), but there is no trustworthy training data: jina-code saw the APPS-train labels, the repository queries are
  few and exposed, and 50 queries only resolve large effects (about 0.08 to 0.14 MRR). Suggested order: a one-parameter
  blend of normalised scores in place of rank fusion, then a small linear model on a few scale-free features, and
  LightGBM only with 200 to 300 or more labelled queries across at least two repositories. The learned score fusion in
  `06-learned-fusion.md` is the step from that list that was taken.
- **Cross-encoder or LLM reranker over the top 5 to 10:** advised against. The smallest credible one, Qwen3-Reranker-0.6B,
  scores below its own 0.6B embedder on MTEB-Code in Qwen's published table (73.42 against 75.41), and would cost roughly
  1.5 to 5 s per query on this CPU (extrapolated from the encoders' timings, not measured).
- **Other cheap second stages Opus assessed** (file aggregation, name priors, tiny-unit penalties, diversity, margin
  gates, pseudo-relevance feedback): not worth trying or expected to hurt registration queries; not run.
- **Fresh queries:** every Node-RED query has been seen and BrowserOS was used to choose settings, so a third fresh set is
  needed before any final claim.
- **Two encoders in the repository path:** the earlier one-encoder rule was retired for the repository search (the second
  encoder costs about the sum of both query encodings) and still holds for the official screening path.
