# Integrating AI Agents into Existing Software Tools

This [usecase presentation](https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/IBM%20OpenPages-AI-Agent-Enablement.pdf) demonstrates how to seamlessly integrate AI agents into existing enterprise software platforms to drive automation and enhance user support. It details the architecture and step-by-step methodology for building, deploying and embedding AI agents within IBM OpenPages-a leading Governance, Risk, and Compliance (GRC) solution. These AI agents are designed to help business users manage compliance requests (BPM), streamline workflows and improve operational efficiency.

# Advanced AI Agents

Agents that capture information beyond digital documents are inherently more advanced than those limited to pre-trained datasets or static documents. This is because the majority of the world's knowledge still exists outside of digitized formats. The next generation of AI agents are those that can directly interface with the physical world like Google's Gemini Assistant (Project Astra), but with the precision, reasoning, and reliability of OpenAI’s models. Such agents are best positioned to lead the future of intelligent systems.

## Why AI Agents

Key reasons: Autonomy, Efficiency, Human-AI Collaboration, Next-Gen Adaptability, Personalization, Productivity, Reasoning, Speed

<img width="795" height="449" alt="image" src="https://github.com/user-attachments/assets/45fc8d2b-31b4-49b6-b8c1-74d11cac49a0" />
Ref: IBM Agentic AI Training 

# Connect 2 AI Agents 

Two AI Agents chat with each other using LLaMA 3.1 Models on separate GPUs.

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/connect_2_ai_agents.py

# Text-to-SQL Agent 

This agent lets users query SQL databases using natural language; it converts questions into SQL, executing them on database, and fetches results.

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/text2SQL_agent.py 

Frameworks: LangChain & OpenAI

# Medical Chatbot Agent

A LoRA-fine-tuned DeepSeek R1 model on medical data to power intelligent medical dialogue systems.

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/medical_chatbot_agent.py 

Method: LoRA (Low-Rank Adaptation)

Frameworks:
```
Unsloth (for efficient fine-tuning)
Hugging Face (Transformers, Datasets)
PyTorch (custom logic)
Weights & Biases (experiment tracking)
Kaggle Notebooks (free GPU) 
```
Instructions:
```
Activate GPU in Kaggle
1. Enable GPU in Kaggle settings  
2. Get API tokens (W&B, Hugging Face)  
3. Add them securely to Kaggle Secrets Manager  
```

# User Reviews (Text/Sentiments) Classifier Agent

A fine-tuned LLM (e.g., DistilBERT) for sentiment classification of user reviews

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/user_review_text_classifier_agent.py 

Fine-Tuning Overview:
```
Fine-Tuning adjusts internal parameters (weights/biases) of a pre-trained LLM to specialize it for a specific task.

For Example: GPT-3 → text-DaVinci-003 (instruction-aligned)

Base vs Fine-Tuned:
Base Model (e.g., GPT-3): General-purpose completions
Fine-Tuned Model (e.g., InstructGPT): Instruction-following and task-optimized

Smaller fine-tuned models (e.g., 1.3B InstructGPT) can outperform larger base models (175B GPT-3) on task-specific benchmarks.
```

3 Fine-Tuning Types:
1. Self-Supervised: Predict next token using raw text
2. Supervised: Learn from labeled input-output pairs
```
1. Choose task
2. Prepare dataset
3. Select base model
4. Fine-tune
5. Evaluate
```
3. Reinforcement Learning: Optimize behavior using human feedback (reward model, PPO fine-tuning)

Parameter Update Strategies:

1. Full Training: Update all model weights
2. Transfer Learning: Tune only final layers
3. PEFT (e.g., LoRA): Freeze base weights, inject small trainable layers
LoRA (Low-Rank Adaptation): Dramatically reduces trainable parameters (e.g., 1M → 4K), improving efficiency

Example:
Model: distilbert-base-uncased
Task: Binary sentiment classification
Steps: Tokenization, padding, accuracy metric

Pre-Tuning: Base model perdorms ~50% accuracy (random chance)
Post-Tuning: Improved training accuracy, slight overfitting observed, better real-world performance

# Research Paper Code replication Agent

An autonomous agent that reads AI research papers, writes code, replicates experiments, and evaluates results — moving towards AI improving AI (Intelligence Explosion)

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/research_paper_code_replication_agent.py.py 

Vision: If AI can read, understand, code, test, and evaluate research, we’re progressing toward self-improving AI systems—a core concept in reinforcement-driven machine learning acceleration. An AI Agent is an autonomous system that perceives its environment, processes information, and takes actions to achieve specific goals. In AI research, these agents can read papers, write code, run experiments, and even innovate.

Research Replication Flow: How AI Agents Conduct AI Research (4-Step Process)
1. Agent Submission
Receives paper (e.g., OpenAI's PaperBench: https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf)
2. Reproduction Execution
Agent writes and runs the experimental code.
3. Automated Grading
Evaluation by GPT-4 or another LLM https://github.com/google/automl 
4. Performance Analysis
Evaluates if agents can replicate and improve research or innovation.

**Benchmarking Agentic AI's**

Agents built with GPT-4.1 outperform GPT-4.5 and other models in terms of hallucination-free performance on shared docs  

https://platform.openai.com/docs/guides/agents 
https://openai.github.io/openai-agents-python/
https://github.com/openai/openai-agents-python



## References 

# Machine Learning
- https://www.coursera.org/learn/machine-learning
- https://see.stanford.edu/Course/CS229
- https://www.datacamp.com/tracks/r-programming
- https://www.datacamp.com/tracks/python-programming
