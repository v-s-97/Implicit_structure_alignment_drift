# Implicit Structure Overrides Explicit Alignment

**Tracing alignment drift in social data trained LLMs**

---

## Overview

This repository accompanies the project **“Implicit Structure Overrides Explicit Alignment: Traceable Drift in Social-Media Trained LLMs”**, developed as part of the *Advanced Machine Learning (AML) 2025/26* course.

We study how **instruction tuning on real-world social-media conversations** reshapes model behavior beyond standard alignment assumptions. In particular, we show that **implicit conversational structure**—such as multi-party, narrative, and socially embedded interactions—can override explicit safety alignment and induce systematic behavioral drift.

---

## Research Question

> Does instruction tuning on social-media conversations induce measurable alignment drift in large language models?

---

## Key Findings

- Instruction tuning on **GrokSet** does **not** improve general assistant capability (MT-Bench).
- It **reduces refusal robustness** under narrative jailbreaks (StrongReject).
- It **increases instrumental and manipulative behavior** in interactive environments (Machiavelli).
- These effects **cannot be explained by explicit framing cues alone**.
- Instead, **implicit social and interactional structure** emerges as a primary driver of alignment drift.

---

## Experimental Setup

- **Base model:** LLaMA-2-7B Chat
- **Instruction-tuned variants:**
  - **WildLlama:** fine-tuned on WildChat (direct user–assistant logs)
  - **GrokLlama:** fine-tuned on GrokSet (public, multi-party social-media conversations)
- **Training:** identical Vicuna-style recipe, same token budget and epochs

---

## Benchmarks

| Benchmark | What it measures |
|---------|------------------|
| **MT-Bench** | General instruction-following capability |
| **StrongReject** | Safety, refusal behavior, jailbreak robustness |
| **Machiavelli** | Goal-directed agency vs ethical constraints |

---

## Results Summary

- **MT-Bench:** GrokLlama underperforms both LLaMA-2 and WildLlama
- **StrongReject:** GrokLlama shows a collapse in refusal robustness under narrative attacks
- **Machiavelli:** GrokLlama exhibits increased manipulation and instrumental strategies

These results indicate a **behavioral regime shift**, not a uniform improvement or degradation.

---


## Content Warning

This repository includes evaluations involving **harmful or sensitive prompts** (e.g. violence, illegal activity, harassment).  
They are included **solely for research and safety evaluation purposes**.

---

## Authors

- **Luca Moresca**
- **Valerio Santini**
- **Nicholas Suozzi**
