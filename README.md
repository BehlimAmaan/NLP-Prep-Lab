# 🧠 NLP Practice Repository

This repository contains structured practice, implementations, and experiments for **Natural Language Processing (NLP)** concepts — covering fundamentals to advanced transformer-based models.

The goal of this repo is:

* Build strong conceptual clarity
* Implement algorithms from scratch where possible
* Apply industry-relevant NLP pipelines
* Prepare for ML/NLP interviews
* Create portfolio-ready projects

---

# 📌 Table of Contents

1. Introduction to NLP
2. Text Preprocessing
3. Feature Engineering
4. Classical ML for NLP
5. Word Embeddings
6. Deep Learning for NLP
7. Transformers & LLMs
8. NLP Projects
9. Evaluation Metrics
10. Interview Preparation Notes

---

# 1️⃣ Introduction to NLP

Natural Language Processing (NLP) is a field of AI that focuses on enabling machines to understand, interpret, and generate human language.

### Key NLP Tasks

* Text Classification
* Sentiment Analysis
* Named Entity Recognition (NER)
* Machine Translation
* Question Answering
* Text Summarization
* Chatbots

---

# 2️⃣ Text Preprocessing

Preprocessing is critical. 70% of NLP pipeline work happens here.

## 🔹 Steps

* Lowercasing
* Removing punctuation
* Removing stopwords
* Tokenization
* Lemmatization / Stemming
* Handling emojis
* Handling URLs, hashtags
* Handling contractions

## 🔹 Tokenization

Splitting text into smaller units (tokens).

Example:

```
"I love NLP!"
→ ["I", "love", "NLP"]
```

Libraries:

* NLTK
* spaCy
* HuggingFace Tokenizers

---

## 🔹 Stemming vs Lemmatization

| Stemming                  | Lemmatization        |
| ------------------------- | -------------------- |
| Rule-based                | Vocabulary-based     |
| Faster                    | More accurate        |
| May produce invalid words | Produces valid words |

Example:

```
running → run (lemma)
running → runn (stem)
```

---

# 3️⃣ Feature Engineering

Before deep learning, features were manually engineered.

## 🔹 Bag of Words (BoW)

Represents text as word frequency.

Example:

```
Text1: I love NLP
Text2: I love ML
```

Vocabulary:

```
[I, love, NLP, ML]
```

Vectors:

```
[1,1,1,0]
[1,1,0,1]
```

Limitation:

* No semantic meaning
* Sparse vectors

---

## 🔹 TF-IDF

TF-IDF = Term Frequency × Inverse Document Frequency

Helps reduce importance of common words.

Formula:

[
TF = \frac{\text{word count in doc}}{\text{total words in doc}}
]

[
IDF = \log\left(\frac{N}{\text{docs containing word}}\right)
]

Used in:

* Search engines
* Ranking systems

---

# 4️⃣ Classical ML for NLP

Common Algorithms:

* Logistic Regression
* Naive Bayes
* SVM
* Random Forest

Pipeline:

```
Text → Preprocess → TF-IDF → ML Model → Prediction
```

Best for:

* Small datasets
* Fast baselines
* Interpretability

---

# 5️⃣ Word Embeddings

Traditional methods ignore semantics.

Word embeddings capture meaning in vector space.

## 🔹 Word2Vec

Models:

* CBOW
* Skip-gram

Concept:
Words appearing in similar contexts have similar vectors.

Example:

```
King - Man + Woman ≈ Queen
```

---

## 🔹 GloVe

Global Vectors for Word Representation
Uses global co-occurrence statistics.

---

## 🔹 FastText

Handles out-of-vocabulary words using subwords.

---

# 6️⃣ Deep Learning for NLP

## 🔹 RNN

Sequential modeling.
Problems:

* Vanishing gradient

## 🔹 LSTM

Solves long-term dependency issue.

## 🔹 GRU

Simpler version of LSTM.

---

# 7️⃣ Transformers & LLMs

Transformers changed NLP completely.

Introduced in:

📄 "Attention Is All You Need" (2017)

Key Concept:
Self-Attention

## 🔹 BERT

* Bidirectional
* Pretrained + Fine-tuned
* Great for classification

## 🔹 GPT

* Autoregressive
* Generative
* Next-word prediction

## 🔹 T5

* Text-to-text framework

---

# 8️⃣ NLP Projects (Practice Section)

### Beginner

* Sentiment Analysis (IMDB dataset)
* Spam Detection
* Fake News Detection

### Intermediate

* NER with spaCy
* Text Summarization
* Topic Modeling (LDA)

### Advanced

* Fine-tune BERT
* Build Question Answering system
* Build Chatbot with Transformers
* RAG system (Retrieval-Augmented Generation)

---

# 9️⃣ Evaluation Metrics

## Classification

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Language Models

* Perplexity
* BLEU Score
* ROUGE Score

---

# 🔟 Interview Preparation Notes

## Common Questions

* Difference between TF-IDF and Word2Vec?
* Why use embeddings over BoW?
* What is attention mechanism?
* How does BERT differ from GPT?
* What is perplexity?
* How do you handle imbalanced text data?

---

# 🛠 Tech Stack Used

* Python
* NumPy
* Pandas
* Scikit-learn
* NLTK
* spaCy
* PyTorch
* TensorFlow
* HuggingFace Transformers

---

# 📂 Suggested Repository Structure

```
NLP-Practice/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│
├── projects/
├── README.md
└── requirements.txt
```

---

# 🚀 Learning Roadmap

1. Text Cleaning
2. BoW + TF-IDF
3. Classical ML
4. Word Embeddings
5. RNN/LSTM
6. Transformers
7. Fine-tuning
8. RAG Systems

---

# 🎯 Goals of This Repository

* Master NLP fundamentals
* Build deployable NLP models
* Prepare for Data Science & ML interviews
* Create production-ready pipelines
* Understand LLM internals


