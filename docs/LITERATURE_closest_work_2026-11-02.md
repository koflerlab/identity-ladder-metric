# Closest published work to the identity ladder (check of 2026-11-02)

Source: a literature comparison run on 2 Nov 2026 over the life-science-at-ML-venues collection (titles and abstracts,
2024 to 2026, 50,185 records with duplicates), followed by first-source checks of the top candidates. Every reference
below was verified with the `bib-verify` procedure (Crossref, OpenAlex, arXiv, publisher pages, Consensus; claims
against abstract or full text). Execution and reproduction status live in `REPRODUCTION_BACKLOG.md` (LS-01 to LS-11,
owned by Codex); this note records why each paper was chosen and what it does and does not show.

## Selection criteria

1. Same question: does image closeness measure preservation of one person's anatomy?
2. Separation of causes: anatomy kept apart from scanner, contrast, noise and other acquisition conditions.
3. Evaluation critique: can a good metric hide the loss of relevant detail?
4. Practical closeness: can the method serve as a comparator, or be tested with the ladder?

The ranking is qualitative, without scores.

## Ranked papers

**1. Wei et al., NeurIPS 2025 [1].** Closest framing of the distance problem. The abstract states that squared
Euclidean distance is not a suitable measure of anatomical distance, because images with the same anatomy need not be
close in it; the authors learn a latent space for OCT harmonization in which the bridge cost follows anatomy. Their
goal is harmonization, ours is scoring individual identity, so their latent space is a candidate to test on the ladder.

**2. Lin et al., NeurIPS 2025 (CARE) [2].** Closest to our critique of RMSE, PSNR and SSIM and to H19. They show that
pixel-wise scores miss small and thin structures and evaluate reconstructions through a frozen multi-organ segmentation
model, with NSD for organs and clDice for vessels and intestines. This is the published version of "evaluate through
downstream segmentation", which H19 questions: a segmentation score can stay high after individual anatomy is lost,
if a mean fill yields a typical segmentation. Our evidence for that is synthetic so far; whether it carries over to
real data and to NSD and clDice (which may behave differently from Dice) is an open experiment, not a result.
Independent review of the score reaggregation: `REVIEW_care_reproduction.md`.

**3. Liu et al., ECCV 2024 (Brain-ID) [3].** Closest to the choice of feature space. Features are trained to be
robust to MR contrast, resolution, orientation and artefacts, with intra- and inter-subject robustness metrics. In
section 4.2 the inter-subject check compares features after registration to a common atlas and treats them as
structurally similar across subjects, so it rewards similarity between people, not discrimination between them. The
paper also describes its features as distinctive to each subject, which is the claim the ladder can test: does
individual detail survive the invariance?

**4. Cohen et al., NeurIPS 2024 [4].** Closest theory. It formalises a tradeoff between perceptual quality and
uncertainty in restoration, with super-resolution and inpainting experiments. It supports keeping realism, fidelity
and identity as separate questions; it has no rungs of re-scan, relative and stranger.

**5. Wei et al., NeurIPS 2026 [5].** Recent MRI work in the line of paper 1: each MR domain is mapped into a shared
latent anatomical pseudometric space before translation; reported on OASIS-3. Only the abstract and the lab's
publication list were reachable (no proceedings record or full text yet), so it ranks below the checked papers.

**6. Liu et al., ICCV 2025 (F-Bench) [6].** Closest outside medical imaging. Face generation and restoration are
rated separately on quality, authenticity, identity fidelity and text-image correspondence. Useful for the design
of a human study; human face similarity does not replace calibration on re-scans, scanners and kinship.

**7. Dey et al., ICLR 2025 [7].** Most useful for domain shift. A data engine synthesises highly variable volumes and
contrastive pretraining is made stable against the simulated nuisance variation; evaluated on multimodal registration
and few-shot segmentation. Good registration or segmentation does not by itself show that individual anatomy is kept,
which is what the ladder would test.

Secondary candidates: local diffusion against structural hallucination [8], an anatomical consistency loss for
multi-contrast MR synthesis [9], a wavelet-based distribution distance for image generation [10], and a distributional
consistency loss for inverse problems [11].

## What this means for novelty

"PSNR and SSIM are not enough" and "use anatomically meaningful features" are well covered. The more specific
contribution of the identity ladder is testing metrics for preservation of individual differences relative to
measurement variability, with separate kinship rungs and an independent population-mean control. None of the papers
above combines all of this. That holds for this selection only and is not proof that no such work exists.

Working order: paper 1 as an alternative distance space, paper 2 as the direct comparator for H19, paper 3 as the
check of whether useful invariance removes individuality.

## References

1. Wei S, Remedios SW, Dewey BE, Bian Z, Wang S, Chen J, et al. Optical coherence tomography harmonization with anatomy-guided latent metric Schrödinger bridges. Adv Neural Inf Process Syst. 2025;38. doi:10.52202/085713-0202
2. Lin T, Li X, Zhuang C, Chen Q, Cai Y, Ding K, et al. Are pixel-wise metrics reliable for computerized tomography reconstruction? Adv Neural Inf Process Syst. 2025;38. doi:10.52202/085713-1132
3. Liu P, Puonti O, Hu X, Alexander DC, Iglesias JE. Brain-ID: learning contrast-agnostic anatomical representations for brain imaging. In: Computer Vision – ECCV 2024. Lecture Notes in Computer Science. Cham: Springer Nature Switzerland; 2024. p. 322–40. doi:10.1007/978-3-031-73254-6_19
4. Cohen R, Kligvasser I, Rivlin E, Freedman D. Looks too good to be true: an information-theoretic analysis of hallucinations in generative restoration models. Adv Neural Inf Process Syst. 2024;37. doi:10.52202/079017-0711
5. Wei S, Remedios SW, Bian Z, Wang S, Zhang J, Chen J, et al. Magnetic resonance unpaired image translation with pseudometric Schrödinger bridges. In: Conference on Neural Information Processing Systems (NeurIPS); 2026. Forthcoming. Available from: https://openreview.net/forum?id=ROSqb47snS
6. Liu L, Duan H, Hu Q, Yang L, Cai C, Ye T, et al. F-Bench: rethinking human preference evaluation metrics for benchmarking face generation, customization, and restoration. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE; 2025. p. 10982–94. doi:10.1109/ICCV51701.2025.01022
7. Dey N, Billot B, Wong HE, Wang CJ, Ren M, Grant PE, et al. Learning general-purpose biomedical volume representations using randomized synthesis. In: International Conference on Learning Representations (ICLR); 2025. Available from: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4f693c15f189efd888b6782a5f4eccb1-Abstract-Conference.html
8. Kim S, Jin C, Diethe T, Figini M, Tregidgo HFJ, Mullokandov A, et al. Tackling structural hallucination in image translation with local diffusion. In: Computer Vision – ECCV 2024. Lecture Notes in Computer Science. Cham: Springer Nature Switzerland; 2024. p. 87–103. doi:10.1007/978-3-031-73004-7_6
9. Shin Y, Lee Y, Jang H, Son G, Kim H, Hwang D. Anatomical consistency and adaptive prior-informed transformation for multi-contrast MR image synthesis via diffusion model. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2025. p. 30918–27. doi:10.1109/CVPR52734.2025.02879
10. Veeramacheneni L, Wolter M, Kuehne H, Gall J. Fréchet wavelet distance: a domain-agnostic metric for image generation. In: International Conference on Learning Representations (ICLR); 2025. Available from: https://proceedings.iclr.cc/paper_files/paper/2025/hash/6605c7d1c1ae1521b328d036df5be1ae-Abstract-Conference.html
11. Webber G, Reader A. Distributional consistency loss: beyond pointwise data terms in inverse problems. In: International Conference on Learning Representations (ICLR); 2026. Available from: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html

## Not fully confirmed

- [2] Title differs by version: the NeurIPS proceedings and Crossref give "...for computerized tomography reconstruction?"; arXiv 2506.02093 gives "...for sparse-view computed tomography reconstruction?". The list uses the version of record; `PROTOCOL_care_reproduction.md` uses the arXiv wording.
- [3], [8] LNCS volume numbers not confirmed by a database, so omitted.
- [5] Venue and acceptance rest on the authors' lab page and one mirror of the OpenReview record; no proceedings record, DOI or arXiv; full text not read; not in Consensus.
- [6], [9] CVF pages returned 403; metadata from Crossref and OpenAlex, which agree.
- [7], [10] No DOI; OpenReview not reachable, ICLR proceedings URL used.
- [1], [2], [4] Crossref page ranges for NeurIPS are indexer artefacts and are left out.
