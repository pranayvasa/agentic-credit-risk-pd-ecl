<!-- prompt: doc_section, version: v1 -->
You are writing one section of a model risk document for a bank's model risk committee.

Document: {{DOCUMENT}} — {{PURPOSE}}
Section: {{SECTION}}
What to cover: {{GUIDANCE}}

Rules:
- Write clear, professional prose in Markdown paragraphs (no headings, no tables; tables are
  added separately by code). 80 to 220 words.
- Use ONLY numbers that appear in the facts below, copied exactly with the same rounding, or
  rounded to fewer decimals. Never compute, estimate or invent a number. Every number you
  write is checked automatically, and unverifiable numbers are flagged.
- Report Gini together with AUC where you mention discrimination.
- Be candid about limitations: the OOT holdout is a random (pseudo) holdout, not temporal; the
  ECL is an illustrative simulation; the target is not the regulatory default definition.
- Do not contradict the validation findings or the gate outcome.

Facts (JSON):
{{FACTS}}
