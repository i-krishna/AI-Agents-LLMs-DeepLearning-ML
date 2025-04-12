# Steps to Fine-tune a pre-trained LLM (HuggingFace)

- Fine-Tuning adjusts internal parameters (weights/biases) of a pre-trained LLM to specialize it for a specific task (e.g., GPT-3 → ChatGPT).

### Base vs. Fine-Tuned Models:

Base (e.g., GPT-3): General-purpose text completion. 
Fine-Tuned (e.g., text-DaVinci-003): Task-aligned and more practical. 

### Smaller Fine-Tuned > Larger Base:
Example: 1.3B InstructGPT outperforms 175B GPT-3 on instruction tasks

### Three Fine-Tuning Methods:

1. Self-Supervised Learning: Predict next token using curated text

2. Supervised Learning: Train on labeled input-output pairs

3. Reinforcement Learning: Based on Human feedback → reward model → PPO fine-tuning

## Fine-Tuning Workflow (Supervised):

1. Choose task

2. Prepare dataset - https://github.com/i-krishna/AI-Agents_LLMs/blob/main/fine-tune-llm.py#L22 

3. Select base model

4. Fine-tune

5. Evaluate

### Parameter Update Strategies:

1. Full Training: Update all model weights

2. Transfer Learning: Update final layers only

3. PEFT (e.g., LoRA): Freeze base weights, inject small trainable layers

### LoRA (Low-Rank Adaptation):
Dramatically reduces trainable parameters (e.g., 1M → 4K), improving efficiency

Example – DistilBERT Sentiment Classifier:

Model: distilbert-base-uncased

Task: Binary sentiment classification

Steps: Tokenization, formatting, padding, accuracy metric

## Pre-Fine-Tuning Evaluation:
Base model performs ~50% accuracy (random chance)

## Post-Fine-Tuning Observations:
Training accuracy improves; some overfitting observed. 
Slight improvement in real-world sentiment prediction




#  Intelligence Explosion & AI Agents 

If AI can Read papers, Understand them, Code and test them, and Evaluate results…

Then we're heading toward AI improving AI (Reinforcement Machine Learning), which could accelerate innovation at a pace faster than humans alone can achieve.

An AI Agent is an autonomous system that perceives its environment, processes information, and takes actions to achieve specific goals. In AI research, these agents can read papers, write code, run experiments, and even innovate.

# Research Paper Replication

[ai_agent_research_replication](/ai_agent_researchpaper_replication.py)

How AI Agents Conduct AI Research (4-Step Process)

1. Agent Submission

Given research papers to replicate (e.g., OpenAI's PaperBench https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf).

2. Reproduction Execution

Develops codebases to reproduce paper results.

3. Automated Grading

An LLM (e.g., GPT-4, https://github.com/google/automl) judges replication accuracy.

4. Performance Analysis

Evaluates if agents can replicate and improve research. 

# Text classification with LLM Models

AI-Agents_LLMs/chat.py 
AI-Agents_LLMs/agents.py
