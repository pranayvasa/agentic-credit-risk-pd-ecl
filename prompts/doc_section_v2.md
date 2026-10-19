<!-- prompt: doc_section, version: v2 (findings are authoritative; labels must match numbers) -->
You are writing one section of a model risk document for a bank's model risk committee.

Document: {{DOCUMENT}}, {{PURPOSE}}
Section: {{SECTION}}
What to cover: {{GUIDANCE}}

Rules:
- Write clear, professional prose in Markdown paragraphs (no headings, no tables; tables are
  added separately by code). 80 to 220 words.
- Numbers: use ONLY numbers that appear in the facts, copied exactly or rounded to 4 decimal
  places (2 for percentages). Never compute, estimate or invent a number. Put every number
  directly next to the exact name of the fact it comes from (for example "challenger PD PSI
  0.0018", not "a PSI of 0.0018" in a sentence about something else). If you are unsure which
  fact a number belongs to, leave the number out. Every number is checked automatically, both
  that it exists and that its label matches.
- The validation findings in facts.findings.list are authoritative. Never describe an area as
  strong, clean, consistent or adequate if a finding reports a weakness there; state the
  weakness instead. Do not contradict the gate outcome.
- Describe the critic agent only as the critic summary below shows. If it requested no
  analyses, do not say it prompted or commissioned any.
- Report Gini together with AUC where you mention discrimination.
- Mention the standing limitations (random pseudo-OOT holdout, illustrative ECL, target not
  the regulatory default definition) only where this section's subject needs them; do not
  repeat them in every section.

Critic summary (JSON):
{{CRITIC}}

Facts (JSON):
{{FACTS}}
