# LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning

<!-- Optional badges (add links) -->
<!-- [![Paper](https://img.shields.io/badge/Paper-EMNLP-blue)](LINK_TO_PAPER) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](LINK_TO_ARXIV) -->
[![arXiv](https://img.shields.io/badge/arXiv-2507.20999-b31b1b.svg)](https://arxiv.org/abs/2507.20999)

**TL;DR.** We introduce **LoRA-PAR**, a dual-system PEFT framework inspired by *Thinking, Fast and Slow*. We (1) **split data** into System-1 (fast, intuitive) vs. System-2 (slow, multi-step) via **multi-model role-play + voting**, (2) compute **element-wise LoRA importance** to **partition a subregion** of parameters for each system (with a shared overlap), and (3) run **two-stage training**: **SFT** on System-1 then **RL (GRPO)** on System-2. With **~40% active LoRA parameters**, LoRA-PAR matches or surpasses strong PEFT baselines on GSM8K, MMLU, and HumanEval.

![main](assets/main.png)


---

## Table of Contents
- [Highlights](#highlights)
- [Overview](#overview)
- [Data Preparation: System-1 / System-2 Labeling](#data-preparation-system1--system2-labeling)
- [Parameter Importance & Subregion Partitioning](#parameter-importance--subregion-partitioning)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Ablations](#ablations)
- [Additional Experiments](#additional-experiments)
- [Limitations](#limitations)
- [Reproducibility & Footprint](#reproducibility--footprint)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Highlights
- **Dual-System PEFT**: Explicitly aligns **data** and **LoRA parameter subregions** with System-1 vs. System-2 demands.
- **Two-stage training**: **SFT → RL (GRPO)** to first consolidate fast, direct mappings, then reinforce multi-step reasoning.
- **Element-wise importance**: Importance via masked loss + 2nd-order proxy selects only the most impactful LoRA **elements**.
- **Efficient yet strong**: With ~**30–40%** active LoRA parameters (at \(\theta \approx 0.9\)), we match or outperform full baselines.

---

## Overview

**Intuition.** Different “subregions” of an LLM’s parameters are more helpful for **fast vs. slow** reasoning. We split both **data** (S1 vs. S2) and **parameters** (S1-only, S2-only, shared) and fine-tune them **selectively**.

**Pipeline.**
1. **Sample Splitter:** Multi-LLM **role-play + voting** labels each example as **System-1** or **System-2**.  
2. **Coordinator:** Compute **element-wise LoRA importance** per system; select top-importance **subregions** (plus shared overlap).  
3. **Two-stage Fine-Tuning:** **Stage-1 SFT** on S1 to strengthen intuition; **Stage-2 RL (GRPO)** on S2 to reinforce multi-step CoT.


> ![main](assets/workflow.png)
> Caption: *Overview of LoRA-PAR: role-play voting → importance-based parameter partitioning → SFT→RL pipeline.*

> ![main](assets/main.png)
> Caption: *Introducing System-1/2 into LLM training and how parameters/data are split and trained.*

---

## Data Preparation: System-1 / System-2 Labeling

We use multi-model role-play + voting to decide whether each instance is S1 (fast, direct) or S2 (multi-step reasoning). Teachers “play” the target model and vote.
