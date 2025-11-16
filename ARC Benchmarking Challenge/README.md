# Benchmarking Challenge for Algoverse AI Safety Fellowship

This repository contains the tasks for the benchmarking challenge as part of the Algoverse AI Safety Fellowship. The goal is to evaluate, compare, and analyze the performance of GPT-4.1-nano and GPT-4o-mini models on the ARC-Challenge dataset using various methods and strategies.

## Evaluate GPT-4.1-nano on the ARC-Challenge Dataset

*Use EleutherAI's LM Evaluation Harness or UK AISI's Inspect to evaluate GPT-4.1-nano on the 2,590 question ARC-Challenge dataset (or some smaller subset of it to save time & tokens).*

The code to implement this is in the `arc.py` file

## Compare GPT-4.1-nano and GPT-4o-mini

*Compare GPT-4.1-nano and GPT-4o-mini on the benchmark and see which has superior performance. Then, do a paired analysis to find the questions that only one of the models get correct. Summarize in which ways one LLM is smarter than the other. Then try to spot quality errors in the questions both LLMs get wrong (are the questions messed up in a way that even a super intelligent model or human can't get them right?).*

The code for this is in the `compare_performance.py` file

Result obtained:
- GPT-4o Mini: accuracy: 0.895 stderr: 0.021
- GPT-4.1 Nano: accuracy: 0.855 stderr: 0.025

GPT-4o Mini slightly outperforms GPT-4.1 Nano, but the difference is not was not statistically significant. 

On the other hand, while some questions were more difficult for one model compared to the other, in most cases, both struggled with similar question types, such as those requiring a mix of factual knowledge and logical analysis. In the paired analysis of the questions only one models gets correct, the main difference is that GPT 4.1 Nano seems to have a harder time with biology questions such as the ones about ecosystems.

Analyzing the questions both got wrong, one that stood out was the following:

> **Question:**
Which of the following will best display percentages of the eight most abundant elements in the Earth's crust?
A. line graph
B. pie chart
C. bar graph
D. data table
CORRECT ANSWER: D
>

This question seems a bit confusing since graphs and data tables serve very different purposes when it comes to visualizing data. Additionally, when searching online the percentage of the most abundant elements in the Earth's crust, the results include many pie charts (such as [this](https://assets.weforum.org/editor/ObkGCyh_XLf5qo0I3MMCdD7Nfi5EURh71jzmyBR3pvs.jpeg)).

## Log-Probability and Model API Limitations

*Can you work around the package's limitations on using log probs on GPT models to obtain a log-probs based evaluation on the ARC-Challenge dataset? For a logprobs based eval we will want to have the model return a set of alternative answers along with the log probs of those answers. This task is intended to serve as a basis for being able to understand options for measuring model output rather than developing a meaningfully different eval in itself. 
First, try picking the most likely answer (looking at the logprobs of the various answer tokens like A, B, C, or D), which is essentially greedy decoding or using temperature=0. Next, use logprobs as a proxy for model confidence and measure whether a models' confidence in the correct answer improves as you add few-shot examples with correct answers. You may decide to normalize the logprobs over the sum of the 4 potential answer tokens as a proxy for % confidence. If you score a model's correctness using its confidence in the correct answer rather than a k-winner-takes-all, do you see a difference in the relative performance of the two models?
Logprobs are accessible in the GPT API, but lm-evaluation-harness/Inspect do not natively handle them because logprobs were disabled for a time. You might be able to write a replacement for the native GPT support which uses the log-probs endpoint, either by subclassing the existing model handler or building a custom solution.*

Results obtained for GPT-4.1-nano:
- **Greedy decoding**: accuracy: 0.855  stderr: 0.025
- **Logprobs (without few-shot examples)** mean: 0.854  stderr: 0.0239  
- **Logprobs (with 10 few-shot examples)**: mean: 0.864  stderr: 0.0226 

Results obtained for GPT-4o-mini:
- **Greedy decoding**: accuracy: 0.895  stderr: 0.0217
- **Logprobs (without few-shot examples)**: mean: 0.877  stderr: 0.0227   
- **Logprobs (with 10 few-shot examples)**: mean: 0.876  stderr: 0.0227

Conclusions:
- Few-shot samples have a minimal positive effect in the probability assigned to the correct answer in GPT-4.1-nano and almost none in GPT-4o-mini
- Logprobs provide a useful proxy for model confidence, enabling nuanced analysis beyond binary accuracy
 