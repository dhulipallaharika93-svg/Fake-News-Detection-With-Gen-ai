# 🛡️ Fake News Detection Using Generative AI

An AI-powered web application that detects whether a news article is likely to be real or fake. The system accepts news text or a news URL, extracts article content using Playwright, classifies the content using a RoBERTa-based machine learning model, and generates an easy-to-understand explanation using Generative AI.

## 🚀 Features

- 📝 Detect fake news from manually entered news text
- 🔗 Analyze news articles using their URL
- 🎭 Playwright-based web content extraction
- 🤖 RoBERTa-based fake news classification
- 📊 Prediction confidence score
- 🧠 Generative AI-powered explanation
- 🌐 Simple and user-friendly web interface
- ⚡ Flask-based backend

## 🏗️ System Workflow

```text
User
  ↓
News Text / News URL
  ↓
Playwright
  ↓
Article Content Extraction
  ↓
RoBERTa ML Model
  ↓
Fake / Real Prediction
  ↓
Generative AI
  ↓
AI Explanation
  ↓
Final Result
