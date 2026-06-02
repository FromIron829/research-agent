# Stage 2 — Agent Answer Quality Rubric

Each agent answer is scored on **four independent dimensions, 0-3 each (max 12)**. An LLM-as-judge (Opus-class) assigns scores, and is given: the question, its difficulty tier + `key_points`, the agent's answer, and **the source chunks the agent cited** (so accuracy and citation claims can be checked against real text, not the judge's memory).

Scores are integers. The judge must justify each score in one sentence before emitting it, and quote/point to evidence from the provided sources when deducting on accuracy or citation.

---

## 1. Factual Accuracy (0-3)
*Are the substantive claims correct, as verifiable against the cited sources?*

- **0** — Contains fabrications or claims **contradicted** by the cited sources; misrepresents what the papers say.
- **1** — One serious error (wrong mechanism, wrong paper, wrong direction of a result) or several smaller inaccuracies that affect the conclusion.
- **2** — Mostly accurate; only minor imprecision or a small error that doesn't change the takeaway.
- **3** — All substantive claims accurate and consistent with the cited sources; no detectable errors.

> Gap/abstain questions (e.g. h07): correctly stating that no single paper does X is **accurate**; inventing a paper that does is a **0-1**.

## 2. Citation Quality (0-3)
*Is every claim cited, do citations resolve to a real retrieved source, and does that source actually support the claim?*

- **0** — No citations, or citations are fabricated / do not support the claims they attach to.
- **1** — Many claims uncited, or citations frequently point to the wrong paper/page (incl. secondhand sourcing for a claim a primary paper covers).
- **2** — Most claims cited and citations generally support them; a few gaps or one weak/secondhand citation.
- **3** — Every substantive claim carries a citation that resolves to a provided source and genuinely supports it; paper + page are correct.

## 3. Completeness (0-3)
*Does the answer cover what the question asks — including the cross-paper synthesis the tier demands?* (Use `key_points` as a guide, not a checklist; equivalent coverage counts.)

- **0** — Misses the central point or answers a narrower/different question.
- **1** — Addresses part but omits major required aspects or papers.
- **2** — Covers the main points; minor omissions.
- **3** — Fully addresses the question. **The bar scales with tier:** simple = the one paper's mechanism explained well; medium = genuine contrast across the 2-3 papers (not parallel monologues); hard = the cross-cluster synthesis / version chain / gap reasoning, not just a list.

## 4. Coherence (0-3)
*Structure, clarity, concision — independent of correctness.*

- **0** — Disorganized, self-contradictory, or hard to follow.
- **1** — Understandable but poorly structured, padded, or repetitive.
- **2** — Clear and well-organized; minor issues.
- **3** — Leads with a direct answer, logically structured, concise, easy to scan (consistent with the agent's ANSWER FORMAT).

---

## Aggregation & reporting
- Report **per-dimension mean** and **total/12**, broken down **by tier** (simple / medium / hard) — the tier breakdown is the point; a high simple-tier score with a low hard-tier score is the diagnostic signal.
- Flag any answer scoring **0-1 on accuracy or citation** for manual review (these are the groundedness failures the eval exists to catch).
- Dimensions are scored **independently**: a fluent, well-cited answer can still be factually wrong (high coherence/citation, low accuracy). Do not let one halo the others.

## Known limitations (state honestly in results)
- Single judge, same model family as the agent (Opus judging Sonnet) → possible self-preference; mitigate by spot-checking against human scores and optionally a cross-family (GPT) judge.
- `key_points` are designer-authored coverage hints, not exhaustive gold; an answer can be complete via different valid content.
- Judge sees only the agent's **cited** sources, not the full papers → it can verify what was cited, but cannot fully detect a *relevant omission* the agent never retrieved. Completeness partially relies on the judge's own expertise.
