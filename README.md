# 🧠 SOFTDSNL Final Exam — Image Captioning + Sentiment Analysis (Flickr8k)

## 🎯 Objective
In this final exam, you’ll **combine image processing and natural language processing** in a single end-to-end application.  
You will use the **Flickr8k dataset**, which contains 8,000 images with **5 captions each**, to train a **caption generation model** and then perform **sentiment analysis** on the generated captions.

---

## 🧩 Learning Goals
- Integrate **computer vision** and **natural language processing** in one project.
- Build and train a **caption generation model** using CNN + LSTM.
- Apply **sentiment analysis** to the generated captions.
- Deploy your trained models in a **Django REST API**.
- Test model endpoints using **Postman**.

---

## 📦 Dataset
- **Dataset:** [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- It contains:
  - ~8,000 images
  - Each image has 5 human-written captions
- You will:
  1. Preprocess images (resize, normalize)
  2. Tokenize and encode text captions
  3. Train a CNN + LSTM model to predict captions from images
  4. Perform sentiment analysis (positive/neutral/negative) on the generated captions

---

## 🏗️ Project Structure
```
flickr8k_project/
│
├── backend/
│   ├── manage.py
│   ├── flickr8k_api/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── ...
│   ├── image_caption/
│   │   ├── models.py
│   │   ├── views.py
│   │   └── utils/
│   │       ├── preprocess.py
│   │       └── caption_generator.py
│   ├── sentiment/
│   │   ├── views.py
│   │   ├── models.py
│   │   └── analyzer.py
│   └── requirements.txt
│
├── model_training/
│   ├── train_caption_model.py
│   ├── train_sentiment_model.py
│   └── evaluation.ipynb
│
├── data/
│   ├── images/
│   └── captions.txt
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/flickr8k_final.git
cd flickr8k_final
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Download Dataset
Download the Flickr8k dataset from Kaggle and place:
- All images in `/data/images/`
- The captions file in `/data/captions.txt`

---

## 🧠 Model Training

### 🖼️ Image Captioning Model
- Uses **InceptionV3** or **ResNet50** for feature extraction
- LSTM layer to generate captions from encoded features
- Train using the preprocessed dataset

**Training script example:**
```bash
python model_training/train_caption_model.py
```

**Outputs:**
- `caption_model.h5` — Trained caption generator model
- `tokenizer.pkl` — Tokenizer for captions

### 💬 Sentiment Analysis Model
- Use a text classification model (e.g., LSTM or BERT)
- Input: generated captions
- Output: sentiment label (positive / neutral / negative)

**Training script example:**
```bash
python model_training/train_sentiment_model.py
```

**Outputs:**
- `sentiment_model.h5`

---

## 🌐 Django Integration

### Endpoints
| Method | Endpoint | Description |
|--------|-----------|-------------|
| `POST` | `/api/predict_caption/` | Upload an image → Returns generated caption |
| `POST` | `/api/predict_sentiment/` | Send text (caption) → Returns sentiment |
| `POST` | `/api/predict_image_sentiment/` | Upload image → Returns caption + sentiment |

Example `POST /api/predict_image_sentiment/` Response:
```json
{
  "caption": "A dog jumping over a log in the forest",
  "sentiment": "positive"
}
```

---

## 📸 Postman Requirements
Submit 10 Postman screenshots showing:
- **5 different images** and their generated captions
- **Sentiment results** for each caption
- Include **both image + text endpoints**

---

## 📝 Deliverables

1. ✅ **PDF Report:** `SOFTDSNL_Final_Surnames.pdf`
   - Model training results (loss/accuracy screenshots)
   - Postman screenshots (10 total)
   - Reflection: Describe what worked and what didn’t
   - GitHub repository link

2. ✅ **GitHub Repository:**
   - All source code (Django + training)
   - README.md (this file)
   - Saved model files

---

## 🧮 Grading Criteria (100 pts)

| Criteria | Description | Points |
|----------|--------------|--------|
| Model Training | CNN + LSTM model correctly trained on Flickr8k | 25 |
| Sentiment Analysis | Functional model for classifying caption sentiment | 15 |
| Django Integration | Functional API endpoints with JSON responses | 25 |
| Postman Tests | Clear and complete test screenshots | 15 |
| Report | PDF includes analysis, results, and reflections | 10 |
| Code Quality | Readability, comments, and proper structure | 10 |


---

## 🧑‍🏫 Submission
Submit your:
- **PDF report**
  - **GitHub link**
  - **Postman screenshots**
  - **Enoch Table**
