# Product Requirements Document (PRD)

## 1. Project Overview
**Project:** LSTM-Based Next-Word Prediction and Text Generation System

A Generative AI application using a PyTorch LSTM language model to learn patterns from a large English text corpus and perform next-word prediction and autoregressive text generation.

The primary capability is **interactive next-word prediction/autocomplete**. Text generation is secondary.

## 2. Problem Statement
Given a sequence of words, the system should predict the most probable next word.

Example:
```text
Input:  once upon a
Output: time
```

The system should also provide multiple candidate predictions and allow the user to continue the text interactively.

## 3. Goals
### Primary
Build a working LSTM language model for next-word prediction.

### Secondary
- Interactive autocomplete
- Autoregressive text generation
- Greedy, temperature, Top-K and Top-P generation
- FastAPI REST API
- Next.js frontend
- Reproducible data, training and evaluation pipeline

## 4. Scope
### In Scope
- English text corpus preparation
- Cleaning and deduplication
- Tokenizer and vocabulary
- LSTM language-model training
- Validation and held-out testing
- Next-word prediction
- Text generation
- FastAPI inference
- Next.js frontend
- Documentation and reproducibility

### Out of Scope
- Chatbot functionality
- General-purpose question answering
- External LLM APIs
- Web search during generation
- Database-dependent core inference
- Foundation-scale language modeling

## 5. Technology Stack
- Python
- PyTorch
- NumPy
- FastAPI
- Pydantic
- Uvicorn
- Next.js
- TypeScript
- Tailwind CSS

## 6. Functional Requirements
- **FR-01:** Accept text input.
- **FR-02:** Tokenize input.
- **FR-03:** Convert tokens to IDs.
- **FR-04:** Predict the next token.
- **FR-05:** Return a configurable number of candidate predictions.
- **FR-06:** Allow interactive autocomplete.
- **FR-07:** Generate text autoregressively.
- **FR-08:** Support generation controls.
- **FR-09:** Expose inference through REST endpoints.
- **FR-10:** Allow the frontend to consume the backend API.
- **FR-11:** Handle unknown tokens.
- **FR-12:** Do not display padding/control tokens as normal autocomplete suggestions.
- **FR-13:** Permit EOS to terminate generated text naturally.

## 7. Non-Functional Requirements
- Reproducible preprocessing and configuration.
- Separation of training and inference.
- No external AI service required for inference.
- Model weights excluded from source control.
- Meaningful API errors.
- Usable desktop UI.
- Held-out evaluation.
- Versioned configuration and metadata.

## 8. Evaluation
Use:
- Cross-entropy loss
- Perplexity
- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy

Generation should also be reviewed for coherence, continuity, repetition and sentence structure.

## 9. Completion Criteria
The project is complete when:
1. The dataset can be prepared reproducibly.
2. The tokenizer can be generated and reused.
3. The LSTM can be trained.
4. Validation metrics can be produced.
5. A best checkpoint can be selected using validation data.
6. The held-out test set can be evaluated.
7. FastAPI can load the selected checkpoint.
8. Next.js can consume the API.
9. Interactive next-word prediction works.
10. Text generation works.
11. Setup and operation are documented.
