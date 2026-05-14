# Part 2 Graph Summary

Generated Part 2 report graphs from Modal metrics plus local judged results.

## Judged results used
- GRPO rm100: 0.6727
- GRPO rm200: 0.6415
- GRPO rm350: 0.7400
- GRPO rm445: 0.7674
- GRPO ReMax rm445: 0.6230
- GSPO+GOPO ensemble min: 0.7000
- GSPO+GOPO ensemble pessimistic: 0.6923
- GSPO+GOPO ensemble mean: 0.8148

## Key takeaways
- Best Part 2 online result: GSPO+GOPO ensemble mean (0.8148)
- Best GRPO-family reward-checkpoint choice: rm445 (0.7674)
- ReMax underperformed the GRPO rm445 baseline (0.6230 vs 0.7674)

## Output files
- part2_judged_win_rates.png
- part2_rm_checkpoint_sweep.png
- part2_grpo_variant_internal_metrics.png
- part2_internal_metrics.csv
