# Literature Review

## Carlo
- Profile GPU, unload unused tokens in vocab
- % pixels correct high after SFT.
- Ideas: steep smooth reward?
- Search next tokens until $p < \epsilon$, not beams $|B| < n$. But discarded, not GPU optimized.
- Ensemble vote: best prob > most frequent.

## [MIT](https://github.com/ekinakyurek/marc)
- Permute transforms.
- ICL in TTFT, leave-1-out. Then auxillary loss on context grids ($y_{n \geq 2}$).
- New LorRA per test task. Reset after.
- Ensemble (vote is all-gather) -- sample $n \times |\tau|$, $n$ = leave-1-out orderings, $|\tau|$ = geometric permutations
- 2 round vote: 1/ each $t \in \tau$ (geometric), 2/ all $t$

## [reARC / ARC-DSL](https://github.com/michaelhodel/re-arc)
- DSL, syntax, generator & verifier functions
- Verifier autochecks test-case output. No NLP explanations, can't remix new tasks -- LLMs.
- Have verifier for 400 train tasks, 400 [public eval](https://arc.net/l/quote/xmesbjnb) tasks, no private eval. DSL design iterated with task solvers, so overfit to trainset probably.
- DSL likely flexible, each private task can be represented by DSL syntax but lengthy or break principles (Python loops, if branches, etc).
- Generator does rejection sampling to place objects in grid, so unknown time (timeout possible). Has difficulty knob corr {num objects, grid size, amount of noise, etc}.
- Rare overdetermined tasks, so almost certain that solver for examples is also solver for test input.
- Verifiers superset of solvers. Longer code. 

## [BARC](https://github.com/xu3kev/BARC)
- 400K ARC-Heavy remixes from GPT4o. But noisy.
- Only seeds verified, not gpt remixes.

## [OmniARC](https://ironbar.github.io/arc24/05_Solution_Summary/)
- Many surrogates:
  1. Ex + in -> out (OG)
  2. Ex -> Code (Induction solver)
  3. Input -> Input (learn sampler)
  4. Ex + in + out -> 1[Correct] (Verifier)
  5. Ex + in + outs -> select correct out (Voter)
- Ex -> Out (OG), Input -> Input (Sampler)
Notebook Flow
- Load `Qwen-2.5-0.5B` base, SFT LoRA
- Agument testset via leave-1-out & permutes
- TTFT on LoRA. `fine-tuning.py`
- Merge LoRA. Run `inference.py` for 96 samples.
- Run `voting.py`, ensemble with 2020 program search.

## [Ryan Greenblatt](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt)
- Test-time RL? Think of program that passes examples. Eventually correct.

## MindsAI
- Not published, only podcasts. Many competitors tried to reverse-engineer.
- Fine-tune on synthetic (reARC) and augmented (geo transform) data
- TTFT (MIT ablations)
- AIRV: Augment, Inference, Reverse Augmentation, Vote
- $p$ permutations -> small $k$ samples per $p$ -> $p^{-1}$ -> max freq/prob.

## DeepSeek R1
- R1-Zero from V3 (verified) on GRPO Math/Code rule reward.
- R1 Warm start: Human cleans R1-Zero thought / few-shot LLM with long CoT.
CoT collect
- R1: Rejection smapling from earlier checkpoint, but for me $\pi_{1B} \neq \pi_{r1}$.