# Iteration 1. SFT

_03-10-2025_

## Goal

Bootstrap good baseline for GPRO by SFT edit reasoning traces; output is DSL Python code, reward is executed % correct, sampler follows Jeremy. 

## Motivation

## Development

## Results

## Conclusion

## Next Steps

## TODO

1. **Set up DSL Environment**
   - [x] Clone and set up the reARC/ARC-DSL repository (version 2.3.4 mentioned in your notes)
   - [x] Verify the DSL can solve all 400 training (and eval) tasks

2. **LLM Prompt Sampler Development**
   - [x] Create prompts for Claude/Gemini/DeepSeek-R1 that avoid "leaky prompts" issue you identified
   - [ ] Include DSL type signatures and few-shot examples in prompts
   - [ ] Implement Jeremy's "evolution" approach rather than Ryan's "sample k -> filter by codeexec"
   - [ ] Design prompts to generate focused reasoning that doesn't "blabber" (avoiding the issue you noted with DeepSeek-R1-Llama-70B counting color numbers incorrectly)
   - [ ] Add docstrings to DSL functions since you noted ARC-DSL lacks NLP explanations
   - [ ] Implement execution environment with proper error handling for syntax/runtime errors
   - [ ] Configure the reward function: correct syntax (0.1) + correct shape (0.1) + (% pixels correct)^10 + small penalty for thinking steps

3. **Edit CoT Collection Pipeline**
   - [ ] Start with warm-start prompts until reasoning generation looks "healthy"
   - [ ] For each task:
     - [ ] Generate initial baseline solution
     - [ ] Request edits from LLMs (Claude preferred based on your notes)
     - [ ] Execute each edited solution to calculate reward delta (Δr)
     - [ ] Implement the rejection sampling strategy: reject if Δr ≤ 0
     - [ ] Select top-k (k=3) by max r_delta with tie-breaking to maximize example coverage
   - [ ] Target ~1000 high-quality edit traces (similar to LIMA's approach you referenced)
   - [ ] Store task difficulty metrics in JSON file for curriculum learning

4. **SFT Dataset Preparation**
   - [ ] Clean collected CoT traces (remove repetitive ideas and fluff)
   - [ ] Format as: prompt\<think>CoT\</think>commented Python
   - [ ] Include "from arc_dsl import *" to let DSL functions be implicitly learned
   - [ ] Create HuggingFace dataset with proper train/eval splits
   - [ ] Manually verify samples to avoid hallucination issues you identified

5. **SFT Model Training**
   - [ ] Start with 2.5B parameter base model (as mentioned in your notes)
   - [ ] Implement vanilla SFT using trl and LoRA (consider AWR if you have a good existing implementation)
   - [ ] Train with batch size 4
   - [ ] Evaluate on held-out tasks
   - [ ] Plan for 8B model after poster (as noted in your timeline)