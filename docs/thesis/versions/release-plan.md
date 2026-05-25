# Phased Release Plan

Three versioned drafts sent to supervisor over 7–9 days. Each phase
delivers complete chapters and explicitly waits on feedback before the
next phase locks dependent content.

## Phase 1 — Day 2-3: Introduction + Related Work
**Scope:** Introduction (motivation, problem, RQs, structure,
contributions) and Related Work. All later chapters present as section
headings with placeholders, so the supervisor sees the architecture
without claims that aren't yet defensible.

**Why this is first:** Related Work framing dictates how the Method
connects mechanisms to literature. Locking Method before the supervisor
has reacted to the literature framing risks rework.

**Cover message:**

> Thanks for the feedback — fair on all four points.
>
> I'd been churning through code and experiment design so heavily
> (plus some external life circumstances) that I completely overlooked
> the literature standards. They're a *standard* for a reason.
>
> I drafted the reasoning chain from scratch to improve the narrative
> for a third-party reader, then moved the content into the DKE
> template structure, taking inspiration from the papers as you
> suggested. Both steps actually made composing my thoughts easier.
>
> This version covers Introduction and Related Work. Method and
> Experimental Design come next — I'd like to incorporate any feedback
> on these two chapters before locking those, since the framing of the
> literature shapes how I connect the Method to it.
>
> A few questions for you that I'll send separately once I've
> consolidated them.

## Phase 2 — Day 5-6: Method + Experimental Design
**Scope:** Adds Method and Experimental Design chapters,
incorporating Phase 1 feedback. Results, Discussion, and Conclusions
remain placeholders.

**Why next:** Each RQ now traces its mechanism to a primitive in the
literature
(`functional_call`+`vmap` for RQ1, self-scheduling +
throughput-ratio variable for RQ2, autotune-and-cache for RQ3).
Experimental Design argues each experiment against its RQ explicitly.

**Cover message:**

> Second pass: Method and Experimental Design, incorporating your
> feedback on the literature framing. Each RQ now traces its
> mechanism to a primitive in the literature (functional_call+vmap
> for RQ1, self-scheduling + throughput-ratio variable for RQ2,
> autotune-and-cache for RQ3). Experimental Design argues each
> experiment against its RQ explicitly.
>
> Results and Discussion next — again, happy to incorporate any
> feedback on these chapters before the final pass.

## Phase 3 — Day 8-9: Results + Discussion + final pass
**Scope:** Adds Results, Discussion (baseline-wins as primary findings
alongside optimization wins), Abstract, Conclusion, and a
citation/cross-reference pass.

**Cover message:**

> Final draft. Results across all three RQs are in; Discussion treats
> baseline-wins as primary findings alongside the optimization wins.
> Abstract, Conclusion, and citation/cross-reference pass are done.
>
> Happy to discuss any remaining gaps in person.

## Phase-to-feedback dependency

- Phase 1 feedback → shapes Method (how literature connects to
  mechanisms) and Experimental Design framing
- Phase 2 feedback → shapes how Results are interpreted and what the
  Discussion needs to argue
- Phase 3 → final synthesis; supervisor feedback at this stage is
  polish-level

## Artifacts (LaTeX sources)

- `../main-phase1.tex` — Introduction + Related Work; later sections
  as "WIP" placeholders; bibliography pruned to entries cited in this
  phase
- `../main-phase2.tex` — adds Method + Experimental Design; bibliography
  pruned to entries cited in this phase
- `../main.tex` — canonical, full thesis; serves as Phase 3 source
