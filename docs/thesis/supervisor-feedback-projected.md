# Supervisor Feedback Projected to Current Draft

Source PDF: `docs/thesis-commented.pdf`

Current draft: `docs/thesis/main.pdf`, generated from `docs/thesis/main.tex`

Extracted annotation count: 75

This file projects annotations from the commented earlier PDF onto the current thesis source. Feedback is grouped by current section and TeX file. Open and partly addressed items appear before items already covered by the current draft. Within each group, items are sorted by estimated effort.

## Complexity Key

- XS: fill placeholders, remove notes, rename labels, fix table formatting.
- S: add one sentence or define one term.
- M: rewrite one paragraph or move a short block.
- L: restructure a section, add citations, or revise an experiment explanation.
- XL: depends on final measurements, figures, or a full results and discussion pass.

## `abstract.tex`

### Open

- XS, `abstract.tex:1`: Make sure to fill this in.
  Action: finish the abstract placeholder after final results are available.
- XL, `abstract.tex:1`: Contributions, conclusion, and abstract should spoil the final results.
  Action: after result tables are populated, write the abstract as a compact answer to the three RQs.

## `main.tex`

### Introduction

- S, `main.tex:41`: Supervisor note: remove introduction subsections and use connecting sentences between paragraphs.
  Action: remove introduction subsections in the final version or convert them into flowing paragraphs.
- S, `main.tex:56`: interactive checkpoint comparison? I assume you mean evaluating the model at these points.
  Action: add one phrase that says a session evaluates multiple related checkpoints over the same loss-grid specification.
- M, `main.tex:100`: Contributions should be rephrased in the final version after all experiments are done. What is the end-result that you get out of your experiment?
  Action: rewrite contributions after results are populated. Use final claims and applicability boundaries.
- L, `main.tex:41`: From RQ: This is something different. Mention what this is.
  Action: separate loss-grid computation from interactive checkpoint comparison. State that checkpoint comparison is a motivating use case.
- L, `main.tex:41`: All mechanism do.
  Action: avoid implying that a shared property belongs to only one mechanism, especially surface validation or grid semantics.
- L, `main.tex:61`, `main.tex:76`: I do not see how this couple to RQ1. In general, you can merge B and C, and put the research questions bolded inline.
  Action: fold the RQs into narrative paragraphs and explain why each question follows from the gap.
- M, `main.tex:61`: These are good! Combine this with the introduction and at the end of the related work sections are appropriate.
  Action: move selected gap sentences into the introduction and remove repetition from related work.

#### Already covered by current draft

- S, `main.tex:61`: available CPU? I assume you mean commodity hardware?
  Covered: current draft uses commodity hardware and available CPU cores.
- S, `main.tex:66`: Link these back to the terminology used before.
  Covered: current problem statement links mechanisms to RQ1, RQ2, and RQ3.
- S, `main.tex:78`: Pytorch comes out of nowhere. Make clear where this comes from.
  Covered: methods section explains PyTorch stateless APIs and cites PyTorch documentation.
- S, `main.tex:78`: 2RQ's.
  Covered: current draft has three RQs.
- S, `main.tex:78`: Two RQ's. You can rephrase these. 100/0 and 0/100 load are the classical settings.
  Covered: current draft has three RQs and frames RQ2 against GPU-only and CPU+GPU scheduling.
- S, `main.tex:85`, `methods.tex:146`: Which calibration step is this?
  Covered: current RQ3 and methods define calibration as policy selection before the session.
- S, `main.tex:85`, `methods.tex:148`: Define what a checkpoints session is.
  Covered: current methods define related checkpoints and session timing. Add the introduction phrase above for earlier clarity.
- M, `related-work.tex:89`: Add this earlier.
  Covered: current problem statement states single-machine mechanisms early.

### Discussion

- L, `main.tex:131`, `results.tex:1`: Results are dry and boring. Just the facts. Discussion is where you do deeper analysis and the implications of it.
  Action: keep measurements in `results.tex`. Move explanations of why vmap, compile, scheduler, and calibration behaved as observed into `DISCUSSION`.
- L, `main.tex:131`: RQ1: Algorithm Redesign Applicability.
  Action: convert discussion bullets into prose with explicit answer paragraphs for RQ1, RQ2, and RQ3.
- L, `main.tex:171`: Related work to support hypothesis is nice. Will check these for the final version.
  Action: ensure final discussion claims about ResNet, GRU, MPS, CUDA, and memory architecture have citations or are stated as interpretations of measured data.
- L, `main.tex:183`: Shared busses and memory may still interfere here.
  Action: add final platform interpretation for unified memory versus discrete VRAM, tied to measured CPU task share and speedup.

### Conclusions

- XS, `main.tex:221`: Make sure to split results and discussion.
  Action: keep `results.tex` descriptive. Turn current discussion TODO bullets into prose after final data lands.
- XL, `main.tex:221`: Contributions, conclusion, and abstract should spoil the final results.
  Action: after tables are populated, answer each RQ in the conclusion.

## `related-work.tex`

### Open

- M, `related-work.tex:89`: Integrate this into the respective sections themselves. You have most of the text, it's just not in the right spot and it's not fully integrated into the why.
  Action: move gap sentences from `Where our thesis sits` into the relevant related-work subsections.
- L, `related-work.tex:89`: These approaches do not work for reason X in our case.
  Action: strengthen the reason: distributed, model-parallel, and request-batching systems do not directly answer one-GPU fixed-grid perturbation evaluation.
- M, `related-work.tex:21`: Usually you don't have sections for an initial paper, unless it is the core of your topic.
  Action: current draft labels the course report as non-peer-reviewed. Add general adaptive grid references only if this section remains prominent.

### Already covered by current draft

- S, `related-work.tex:7`: baseline of?
  Covered: current related work names Li et al. and Goldstein implementation as the baseline source.
- S, `related-work.tex:21`: Can remove the A,B,C,D and just keep the title.
  Covered: current draft uses normal subsection titles.
- S, `experiments.tex:89`: Good. There is likely a non-arxiv version of this paper as well.
  Covered: current bibliography cites the published LossLens DOI.
- M, `related-work.tex:1`: Good, but add a sentence to make it flow.
  Covered: current related work has an opening scope sentence.
- M, `related-work.tex:7`: much better start than before! Add a connection to this to the current paper.
  Covered: current `Where our thesis sits` section states the one-machine fixed-grid gap.
- M, `related-work.tex:35`: This has much improved since the latest version.
  Covered: no action.
- M, `related-work.tex:35`, `related-work.tex:89`: Good overview. What is still missing, is how this connect to the thesis itself.
  Covered: current placement section connects heterogeneous scheduling to RQ2 and states which systems are precedents.
- M, `related-work.tex:69`: This needs to be a little clearer, but in general this is what we expect from the connection.
  Covered: current autotuning paragraph maps ATLAS-style calibration to a machine/workload cache key.

## `methods.tex`

### Open

- XS, `methods.tex:9`: Can defer that to later.
  Action: remove the author note that starts `NOTE: these are repeated...` before submission.
- XS, `methods.tex:66`: `+` is a little confusing. Can simply use new-line and add horizontal lines.
  Action: check the rendered table. If cramped, split candidate components over two lines.
- M, `methods.tex:12`: Ensure that you compare this to the results in their paper.
  Action: add a short note in results or limitations comparing baseline scale to Li and Goldstein when final timings are available.
- M, `methods.tex:132`, `experiments.tex:161`: Did you find any related literature that supports this methodology?
  Action: add a citation or methodological justification for artificial slowdown if final text relies on slowed rows.
- M, `methods.tex:132`, `main.tex:183`: Slowing the GPU is indeed much more sensical as the CPU.
  Action: add an explicit verification row or limitation comparing CPU-only against CPU+GPU under large GPU sleep.
- M, `methods.tex:180`: Why this particular formula?
  Action: add the rationale: enough probe tasks to cover GPU-only plus CPU-worker candidates with a small multiple.

### Already covered by current draft

- S, `methods.tex:52`: What is T and s? You have not defined these.
  Covered: current draft defines `T_b`, `T_v`, `T_c`, `T_vc`, and speedups.
- S, `methods.tex:52`: T_v is time for the vanilla implementation?
  Covered: current draft defines baseline-relative speedups.
- S, `methods.tex:116`: What is the canonical loop?
  Covered: current methods define the canonical algorithm before the scheduler.
- S, `methods.tex:132`: not needed.
  Covered: earlier unnecessary detail appears removed or compressed.
- S, `methods.tex:214`: wall?
  Covered: current methods define wall time through `time.perf_counter_ns`.
- S, `methods.tex:217`: What is a session here? Single point? Batch of points?
  Covered: current methods define `T_grid` and `T_session`.
- S, `methods.tex:221`: platform? Workload? What are these? Are these consistent? What are the conditions?
  Covered: current experiments define platform, workload, and condition.
- S, `methods.tex:233`: CI?
  Covered: current methods define confidence-interval verdicts.
- S, `methods.tex:224`: These are not defined. Can use 1 line to describe all of these.
  Covered: current baseline decomposition defines each timing component.
- S, `methods.tex:245`: Make it clear that this is your interpretation of it. Statistically CI's mean something slightly different.
  Covered: current draft calls these timing verdicts and distinguishes RQ3 descriptive outcomes.
- M, `methods.tex:34`: Link back to related work. Why is this expectation there?
  Covered: current RQ1 methods cite PyTorch ensembling and compile docs.
- M, `methods.tex:52`: vanilla or baseline? Stay consistent.
  Covered: current draft mostly uses `vanilla` as the condition and `baseline` as the algorithm. Check final captions for consistency.
- M, `methods.tex:52`: Why not go from the s_v directly? Are you assuming vc is always faster?
  Covered: current draft compares the composed candidate to the better individual transform using `max(s_v,s_c)`.
- M, `methods.tex:83`: Unsure where this is coming from, the enumerate does not seem to connect to previous paragraph.
  Covered: current draft removed the enumerate and uses short method subsections.
- M, `methods.tex:83`: It is not clear to me what you are trying to do here to explain the methods.
  Covered: current draft defines `functional_call`, `vmap`, chunking, and expected speed effect.
- M, `methods.tex:113`: Add in the RQ's at the top such that the reader has no need to go back and forward.
  Covered: current methods opening maps approaches to RQ1 through RQ3.
- M, `methods.tex:113`: Link back to related work again. Should be straightforward.
  Covered: current scheduling section references `rw-scheduling`.
- M, `methods.tex:121`: is it? Is it always the exact same set of instructions?
  Covered: current draft says both devices run the same forward and loss code and share validation. If this is not strictly true at backend level, revise to `same model and loss semantics`.
- M, `methods.tex:132`: I would suggest putting this difference only in at RQ3, or at the very start of the section.
  Covered: current slowdown explanation is localized in RQ2 and RQ3 inherits a selected operating point.
- M, `methods.tex:146`: This should be earlier.
  Covered: calibration now appears in methods before experiments, and the introduction names RQ3.
- M, `methods.tex:146`: Don't need this here.
  Covered: earlier extra material appears condensed. Keep checking final rendered length.
- M, `methods.tex:162`: determined by RQ1?
  Covered: current draft states `rq3_config` is determined by RQ1 for the selected workload/platform pair.
- M, `methods.tex:189`: I do not see how this is amortizing the callibration.
  Covered: current draft defines amortization as one-time upfront calibration whose cost is recovered by per-checkpoint savings and gives break-even `N*`.
- M, `methods.tex:248`: Why would 1 and 2 happen? Are you not always computing the same things?
  Covered: current validation rules frame these as surface equivalence checks.
- M, `methods.tex:252`: Is (3) an issue?
  Covered: current validation rule treats surface mismatch as suppressing timing claims.

## `experiments.tex`

### Open

- XS, `experiments.tex:27`: overflow.
  Action: re-render page 5 after table edits. Fix table overflow, especially platform and workload tables.
- S, `experiments.tex:148`: Been a while since r was used here. Short reintro is usefull.
  Action: add a short reminder at the start of RQ2: `the CPU/GPU throughput ratio r`.
- M, `experiments.tex:20`: Here the exact models are important. Can put this in the appendix if you lack space.
  Action: add exact model definitions, training checkpoints, dataset splits, and asset manifest references in the appendix or experiment setup.
- XL, `experiments.tex:208`, `results.tex:155`: Calibration cache aim and amortization should be clear.
  Action: once RQ2 selects the regime and RQ3 has timings, verify that `rq3_config`, calibration cost, session speedup, break-even `N*`, and the amortization label form one coherent story.

### Already covered by current draft

- S, `experiments.tex:43`: workload? Use consistent terminology.
  Covered: current draft defines workload as a dataset, model family, loss triple.
- M, `experiments.tex:1`: Why are these mixed? Do you mean that you use the output of RQ1 for RQ2, and the output of RQ2 for RQ3?
  Covered: current experiment opening describes the decision pipeline and states baseline comparisons.
- M, `experiments.tex:1`: Mirror this in the text.
  Covered: current text mirrors the pipeline and table captions.
- M, `experiments.tex:17`: good!
  Covered: no action.
- M, `experiments.tex:77`: Argumentation here is good, referring to existing approaches.
  Covered: no action beyond final proofread.
- M, `experiments.tex:145`: Related work to support hypothesis is nice. Will check these for the final version.
  Covered: current RQ2 hypothesis cites self-scheduling and points to platform discussion.
- M, `experiments.tex:183`: Ah, this is what you mean. This is not clear in the methods section.
  Covered: current RQ3 question, motivation, hypothesis, and design state the comparison and break-even condition.
- M, `experiments.tex:183`: B? RQ1 Second experiment? Keep it to 1 name to prevent confusion.
  Covered: current draft uses RQ1, RQ2, RQ3 and Experiment 1, 2, 3 consistently.

## `results.tex`

### Open

- XS, `results.tex:36`, `results.tex:155`: Visualize these results for easier comparison.
  Action: add bar charts or dot plots for final RQ1 and RQ2 tables when numbers are populated. Start axes from zero.
- S, `results.tex:128`: rows?
  Action: clarify in the RQ2 table caption whether rows mean workload/platform rows, native rows, slowed rows, or table rows.
- M, `results.tex:155`: This I do not follow. What is the aim here?
  Action: after final numbers, add one leading sentence before Table `tab:calib` that says RQ3 tests whether one upfront calibration is recovered within `N=4` checkpoints.
- XL, `results.tex:10`, `results.tex:39`, `results.tex:82`, `results.tex:128`, `results.tex:158`: Tables and final experiments need final values.
  Action: populate all placeholder result tables. Final contribution, discussion, abstract, and conclusion text depends on these values.
- XL, `results.tex:36`, `results.tex:125`: Visualize these results for easier comparison.
  Action: generate final figures from measurement artifacts and cite them from results.

## `appendices.tex`

### Open

- M, `experiments.tex:20`: Here the exact models are important. Can put this in the appendix if you lack space.
  Action: if the experiment section becomes too dense, move exact model, checkpoint, dataset, and asset manifest details here.
