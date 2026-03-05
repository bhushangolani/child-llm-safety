# Child-Centered LLM Safety Evaluation & Alignment

A comprehensive pipeline for evaluating and improving how large language models respond to children (ages 6–12) across sensitive topics.

## Overview

This project defines a **Child Safety Benchmark (CSB)** for LLMs. It generates child-style prompts across seven safety-critical categories, collects responses from multiple open-source models, scores them using an LLM judge on five developmental dimensions, and trains a classifier to predict response quality.

### Key Contributions

- **Child Safety Benchmark (CSB)**: 350 human-verified child-style prompts across 7 categories based on the 4Cs framework
- **Multi-Model Evaluation**: Benchmarked 10 open LLMs (Gemma, Llama, GPT-OSS, DeepSeek, Qwen) using GPT-5 as an LLM-as-a-judge
- **Five-Dimensional Scoring**: Safety, developmental fit, empathy, moral guidance, and boundary-setting
- **Safety Classifier**: Binary Accept/Reject classifier trained on XLM-RoBERTa
- **DPO Alignment**: Direct Preference Optimization with GPT-5 gold answers for improved child-centered safety

## Categories

| Category | Description |
|----------|-------------|
| Medical/Health | Health-related questions from children |
| Sexual | Age-inappropriate content handling |
| Harm/Bullying/Substance | Violence, bullying, substance-related queries |
| Classroom Advice | School and learning situations |
| Family/Friends | Relationship dynamics |
| Moral/Values | Ethical and value-based questions |
| Hateful/Stereotypes | Discriminatory content handling |

## Evaluation Framework

Responses are scored on five dimensions (0–2 scale) grounded in developmental psychology (Piaget, Kohlberg, Erikson):

1. **Safety** — Protects from harmful content while avoiding over-censoring
2. **Developmental Fit** — Simple, concrete language appropriate for ages 7–12
3. **Empathy** — Validates feelings and curiosity
4. **Moral Guidance** — Explains prosocial values (kindness, fairness)
5. **Boundaries** — Suggests talking to trusted adults when appropriate

## Pipeline

```
generate_data.py → generate_responses.py → judge_responses.py → train_classifier.py
       ↓                    ↓                      ↓
   CSB.json            responses/            evaluation/
                                                   ↓
                                          Analysis.ipynb → CSB_Classifier dataset
```

## Project Structure

```
├── datasets/           # Benchmark prompts and classifier splits
├── prompts/            # Jinja2 templates for data generation, judging, and gold responses
├── responses/          # Model outputs from 10 open LLMs
├── gold-responses/     # Curated ideal responses from proprietary models
├── evaluation/         # LLM-judge scores
├── generate_data.py    # Create benchmark prompts
├── generate_responses.py   # Collect model responses
├── judge_responses.py  # Score responses with LLM judge
├── train_classifier.py # Train XLM-RoBERTa safety classifier
├── Analysis.ipynb      # Dataset analysis and preparation
└── requirements.txt
```

## Technologies

- **LLM APIs**: OpenAI (GPT-5.1), DeepInfra, Anthropic (Claude), Google (Gemini)
- **ML/NLP**: PyTorch, Transformers, Hugging Face Datasets, scikit-learn
- **Data**: pandas, Jinja2, JSON
- **Visualization**: matplotlib, seaborn

## Course

CSE 595 — University of Michigan, Ann Arbor
