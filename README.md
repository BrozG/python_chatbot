# 🤖 Python Chatbot — My First AI Project

> A rule-based neural network chatbot built with Python and JSON intents — made before LLMs like ChatGPT even existed! Features Google search through the terminal.

> 🏆 **This was my very first AI project** — the beginning of everything.

---

## ✨ Features

- 💬 **Intent-based conversation** — responds based on trained JSON intents
- 🔍 **Google Search** — can search the web directly from terminal
- 🎮 **Text games** — play simple text-based games
- 🕐 **Tell time** — asks for and reports current time
- 🧠 **Custom trained model** — neural network trained on custom intents
- 🐍 Pure Python — no LLM APIs needed

---

## 🧠 How It Works

```
User Input
      ↓
┌─────────────────────────┐
│   Intent Classification │  ← Neural network predicts intent
│   (brozmodel.h5)        │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│   Response Selection    │  ← Picks response from intents.json
└─────────────────────────┘
      ↓
   Response to User
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Neural Network | Keras / TensorFlow (.h5 model) |
| Intent Data | JSON |
| NLP | NLTK |
| Search | Google Search via terminal |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/BrozG/python_chatbot
cd python_chatbot
```

### 2. Install dependencies
```bash
pip install tensorflow nltk
```

### 3. Run the chatbot
```bash
python chatbot/main.py
```

---

## 📁 Project Structure

```
python_chatbot/
├── chatbot/
│   └── main.py              # Main chatbot script
├── intents.json             # Training intents & responses
├── brozmodel.h5             # My custom trained model
├── chatbot_model.h5         # Alternative trained model
├── classes.pkl              # Intent classes
├── words.pkl                # Vocabulary
└── data.pickle              # Training data
```

---

## 📖 Context

This was built **before ChatGPT** — back when building a chatbot meant training your own neural network from scratch on custom data. No GPT API, no LangChain, just pure Python and a self-trained model.

Looking back, I would have used an open source LLM instead of a rule-based approach — but this project taught me the fundamentals of NLP, intent classification, and neural networks that everything else is built on.

---

## 💡 What I'd Do Differently

- 🤖 Use an open source LLM instead of rule-based intents
- 🌐 Add a proper web UI instead of terminal only
- 📊 More training data for better accuracy

---

## 👤 Author

**BrozG** — Full Stack & AI/ML Developer

[![GitHub](https://img.shields.io/badge/GitHub-BrozG-blue)](https://github.com/BrozG)

---

## 📄 License

MIT
