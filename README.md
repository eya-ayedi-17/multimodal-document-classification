# Multimodal Document Classification

A deep learning pipeline for classifying visually similar documents by combining **visual** and **textual** information.

> ⚠️ **Confidentiality notice**
>  Due to a confidentiality agreement, the **source code**, **trained model weights**, and **dataset** cannot be shared publicly. This repository documents the **methodology, architecture, and approach** only.

---

## Context

Classifying documents automatically becomes difficult when several categories share a very similar visual layout (e.g. different types of invoices or medical prescriptions). Relying on the document image alone is often not enough — the textual content carries information that can resolve the ambiguity.

This project builds a pipeline that combines both signals to classify documents into **12 categories**.

## Architecture

![Pipeline architecture](assets/pipeline.png)

The pipeline has two parallel branches that are fused before classification:

**Visual branch**
- Image preprocessing (resize, normalization)
- Feature extraction with a ResNet encoder (ResNet50 / ResNet101)
- 2048-dimensional visual embedding

**Text branch**
- Text extraction with PaddleOCR
- Keyword selection (NLP + KeyBERT)
- Semantic encoding with Sentence-BERT
- 384-dimensional text embedding

**Fusion & classification**
- The two embeddings are concatenated (2432-dim)
- A Multi-Layer Perceptron (MLP) classifies the fused vector into one of 12 document classes

## Key challenges addressed

- Documents with nearly identical visual layouts but different textual content
- OCR errors and incomplete text extraction
- Trade-off between classification accuracy and inference cost
- GPU memory management when running multiple pretrained encoders simultaneously

## Tech stack

`Python` · `PyTorch` · `PaddleOCR` · `Sentence-Transformers` · `KeyBERT` · `ResNet (torchvision)`

## What this repository contains

- ✅ Architecture documentation and diagrams
- ✅ Methodology and design decisions
- ❌ Source code (confidential)
- ❌ Dataset (confidential)
- ❌ Trained model weights (confidential)

## About

This project was carried out as part of my summer internship (2025–2026) at [Tufratech](https://www.tufratech.com), under the supervision of **Bilel Kammoun**, with guidance from **Rima Meziou**.

**Author:** Eya Ayedi — Telecommunications Engineering student, Sup'Com Tunis
Focus: Artificial Intelligence, Machine Learning, Deep Learning
