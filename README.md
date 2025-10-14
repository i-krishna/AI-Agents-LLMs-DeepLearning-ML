# Integrating AI Agents into Existing Software Tools

This [usecase presentation](https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/Business%20Process%20Automation%20with%20AI-Agents.pdf) demonstrates how to seamlessly integrate AI agents into existing enterprise software platforms to drive automation and enhance user support. It details the architecture and step-by-step methodology for building, deploying and embedding AI agents within IBM OpenPages-a leading Governance, Risk, and Compliance (GRC) solution. These AI agents are designed to help business users manage compliance requests (BPM), streamline workflows and improve operational efficiency.

# Medical Chatbot Agent

A LoRA-fine-tuned DeepSeek R1 model on medical data to power intelligent medical dialogue systems. Method: LoRA (Low-Rank Adaptation)

The purpose of this [code](https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/medical_chatbot_agent.py) is to fine-tune a large language model (DeepSeek-R1) for advanced medical reasoning and clinical case analysis. By training the model on a specialized medical dataset using LoRA and Unsloth, it enables the model to generate accurate, step-by-step answers to complex medical questions, making it more effective for healthcare automation and decision support.

Alternatively, you can fine-tune this model using H2O LLM Studio’s no-code graphical user interface (GUI) by following the steps outlined in this [guide](https://github.com/i-krishna/AI-Agents-LLMs-DeepLearning-ML/blob/main/H2O%20LLM%20Studio_Fine%20Tune_No%20Code%20GUI.pdf).

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
# Text-to-SQL Agent 

This agent lets users query SQL databases using natural language; it converts questions into SQL, executing them on database, and fetches results.

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/text2SQL_agent.py 

Frameworks: LangChain & OpenAI

# Connect 2 Connversational Agents 

Two AI Agents chat with each other using LLaMA 3.1 Models on separate GPUs.

Code: https://github.com/i-krishna/AI-Agents_LLMs_DeepLearning_ML/blob/main/AI_Agents%20/connect_2_ai_agents.py

## Connecting Multiple Agents 

<img width="795" height="449" alt="image" src="https://github.com/user-attachments/assets/45fc8d2b-31b4-49b6-b8c1-74d11cac49a0" />
Ref: IBM Agentic AI Training 

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

# Fine-Tuning Types:
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


# Advanced AI Agents

Agents that capture information beyond digital documents are inherently more advanced than those limited to pre-trained datasets or static documents. This is because the majority of the world's knowledge still exists outside of digitized formats. The next generation of AI agents are those that can directly interface with the physical world like Google's Gemini Assistant (Project Astra), but with the precision, reasoning, reliability and explaniability of model behavior. Such agents are best positioned to lead the future of intelligent systems.


**Benchmarking Agentic AI's**

Agents built with DeepSeek Outperform GPT models and GPT-4.1 outperform GPT-4.5 in terms of hallucination-free performance on shared docs  

https://api-docs.deepseek.com/
https://platform.openai.com/docs/guides/agents 
https://openai.github.io/openai-agents-python/
https://github.com/openai/openai-agents-python

Key reasons to Use AI Agents: Autonomy, Efficiency, Human-AI Collaboration, Next-Gen Adaptability, Personalization, Productivity, Reasoning, Speed

# Other Applications 

1. RAG flow for Domain specific tasks: Email Alert for Fraud Detection

<img width="760" height="646" alt="image" src="https://github.com/user-attachments/assets/585b8baf-7a73-45c0-b7f7-73d47848c95b" />

2. LLM Integration into Data Pipeline

Closed Source API-based LLMs (OpenAI)

<img width="1758" height="984" alt="image" src="https://github.com/user-attachments/assets/eef10a0d-6041-4bc5-b437-b1cf14924546" />

Open Source API-based LLMs in HuggingFace Platform

<img width="702" height="448" alt="image" src="https://github.com/user-attachments/assets/cb72d6d1-b641-48b5-b719-06266d02c513" />

3. Classic Deep Learning & Machine Learning

<img width="1690" height="672" alt="image" src="https://github.com/user-attachments/assets/a69a9112-735f-4177-8f23-f2d4c4810b9b" />

<img width="2528" height="962" alt="image" src="https://github.com/user-attachments/assets/b65b07e4-f8dc-4d91-9f45-f98c24978530" />

<img width="1016" height="591" alt="image" src="https://github.com/user-attachments/assets/2203f9c3-9468-4646-b337-c76d895d5a3f" />

<img width="1074" height="705" alt="image" src="https://github.com/user-attachments/assets/3b7aa548-e1a4-4bc9-a103-34dbf12807dc" />

## References 
- Open-source LLMs: www.gpt4all.io, LLaMA, Mistral, BERT
- Closed-source LLMs: OpenAIs GPT, Anthropics Claude 
- https://www.knime.com/ 
- https://www.coursera.org/learn/machine-learning
- https://see.stanford.edu/Course/CS229
- https://www.datacamp.com/tracks/r-programming
- https://www.datacamp.com/tracks/python-programming
