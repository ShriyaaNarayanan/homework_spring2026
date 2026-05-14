# Part 1 Graph Summary

Generated graphs from Modal metrics for the available Part 1-relevant runs.

## Available runs on Modal
- reward model: `wildchat_min4_judged_5k_reward_model_v1`
- offline: `wildchat_min4_judged_5k_aot_beta02_v1`
- online: `wildchat_min4_judged_5k_grpo_rm445_v1`

## Required Part 1 runs not found on Modal
- `wildchat_min4_judged_5k_dpo_beta01_v1`
- `wildchat_min4_judged_5k_ipo_v1`
- `wildchat_min4_judged_5k_drgrpo_rm445_v1`
- `wildchat_min4_judged_5k_gspo_rm445_v1`

## Key numbers
- reward model best eval pair accuracy: `0.8477` at step `325`
- reward model final eval pair accuracy: `0.8359`
- AOT best eval reference-corrected preference accuracy: `0.7852` at step `350`
- AOT final eval reference-corrected preference accuracy: `0.7656`
- GRPO RM445 final eval RM-fraction-policy-above-reference: `0.4062`
- GRPO RM445 final eval RM-margin-policy-minus-reference: `-1.5252`
- GRPO RM445 local judged online score already recorded in `rm445_results.json`: `0.7674`

## Output files
- `reward_model_overview.png`
- `aot_overview.png`
- `grpo_rm445_overview.png`
