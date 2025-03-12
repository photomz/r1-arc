# Iteration 1. SFT

_03-10-2025_

## Goal

Bootstrap good baseline for GPRO by SFT edit reasoning traces; output is DSL Python code, reward is executed % correct, sampler follows Jeremy. 

## Motivation

## Development
- Skip pooling complexity.
- Edit prompt by git diff.
- First dry-run w/o changes.

Evolutionary sampling
- DeepSeek-r1 (not distilled) slowly improves thinking (shapes, grids, boxes, does self-check examples and code well).
- But very slow & thought branches (30 "Alternatively ... " prefixes). 6400 reasoning tokens??

### R1 Prompting
- Roughly, reasoning 50% on correct interp of problem (abstract thinking, sometimes manually check if confident).
- and 50% on how to write gussed pattern in Python (many versions before final program, is obv wrong tho).  
- 11pm: Don't adapt others' repos, too hard to read & bloat. Clean data pipeline from scratch is ok. Code exec, healing is not hard. Steal reasoning CoT string only. 
- 3.7 Sonnet gets DSL {solvers, reused strats, typedefs} with low comments 
- Then next test: 1 shot DSL w Cursor prompt? Quick test, need eval suite to (1) prettify ICL format (2) Simple exec server.
- Allow `print(xk)` to debug most uncertain.
- Notice big degradation w Groq SpecDec. Retire

- [x] Grid format viewer
- [x] LLM providers w/ Typer stub input: grok, grok spec dec, hyperbolic, deepseek china. Async.
- [ ] Exec server

## Results


## Conclusion

## Next Steps

## TODO

1. **Set up DSL Environment**
   - [x] Clone and set up the reARC/ARC-DSL repository (version 2.3.4 mentioned in your notes)
   - [x] Verify the DSL can solve all 400 training (and eval) tasks

2. **LLM Prompt Sampler Development**
   - [x] Create prompts for Claude/Gemini/DeepSeek-R1 that avoid "leaky prompts" issue you identified
   - [x] Dry run prompt sampler
   - [x] Check execution (Bad, 16k tokens for 10m but wrong answer.)
   - [ ] Why won't it parallelize

3. DSL Execution and Prompt
   - [ ] Include DSL type signatures and few-shot examples in prompts
   - [x] Implement Jeremy's "evolution" approach rather than Ryan's "sample k -> filter by codeexec"
   - [ ] Design prompts to generate focused reasoning that doesn't "blabber" (avoiding the issue you noted with DeepSeek-R1-Llama-70B counting color numbers incorrectly)
   - [ ] Add docstrings to DSL functions since you noted ARC-DSL lacks NLP explanations
   - [ ] Implement execution environment with proper error handling for syntax/runtime errors
   - [ ] Configure the reward function: correct syntax (0.1) + correct shape (0.1) + (% pixels correct)^10 + small penalty for thinking steps

4. **Edit CoT Collection Pipeline**
   - [ ] Start with warm-start prompts until reasoning generation looks "healthy"
   - [ ] For each task:
     - [ ] Generate initial baseline solution
     - [ ] Request edits from LLMs (Claude preferred based on your notes)
     - [ ] Execute each edited solution to calculate reward delta (Δr)
     - [ ] Implement the rejection sampling strategy: reject if Δr ≤ 0
     - [ ] Select top-k (k=3) by max r_delta with tie-breaking to maximize example coverage
   - [ ] Target ~1000 high-quality edit traces (similar to LIMA's approach you referenced)
   - [ ] Store task difficulty metrics in JSON file for curriculum learning

5. **SFT Dataset Preparation**
   - [ ] Clean collected CoT traces (remove repetitive ideas and fluff)
   - [ ] Format as: prompt\<think>CoT\</think>commented Python
   - [ ] Include "from arc_dsl import *" to let DSL functions be implicitly learned
   - [ ] Create HuggingFace dataset with proper train/eval splits
   - [ ] Manually verify samples to avoid hallucination issues you identified

6. **SFT Model Training**
   - [ ] Start with 2.5B parameter base model (as mentioned in your notes)
   - [ ] Implement vanilla SFT using trl and LoRA (consider AWR if you have a good existing implementation)
   - [ ] Train with batch size 4
   - [ ] Evaluate on held-out tasks
   - [ ] Plan for 8B model after poster (as noted in your timeline)