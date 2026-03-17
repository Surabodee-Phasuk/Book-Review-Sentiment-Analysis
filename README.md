# 📚 BookInsight — Book Review Sentiment Analysis

> Understanding Reader Emotions through Aspect-Based Sentiment Analysis with by team 3 people

---

## 🔍 Overview

การวิเคราะห์ sentiment แบบทั่วไปบอกได้แค่ว่ารีวิวเป็นบวกหรือลบ แต่ไม่บอกว่า *ส่วนไหน* ของหนังสือที่ถูกชมหรือถูกตำหนิ

โปรเจกต์นี้จึงใช้ **Aspect-Based Sentiment Analysis (ABSA)** เพื่อวิเคราะห์ความรู้สึกของผู้อ่านแยกตามแต่ละด้าน เช่น `plot`, `character`, `storytelling`, `price` และเปรียบเทียบประสิทธิภาพระหว่าง:

- **Baseline** — Logistic Regression (TF-IDF)
- **Deep Learning** — Bidirectional LSTM

---

## 📁 Repository Structure

```
Book-Review-Sentiment-Analysis/
├── Book Review Sentiment Analysis/
│   ├── Aspect_Based_Sentiment_Analysis.ipynb   # Main notebook
│   ├── preprocessed_review.csv                  # Cleaned dataset
│   ├── my_lstm_absa_model.h5                    # Trained BI-LSTM model
│   └── my_baseline_lr_model.joblib              # Trained Logistic Regression model
├── cs374_NLP_BookSentiment Analysis.pdf         # Project presentation
└── README.md
```

---

## 📊 Dataset & Preprocessing

| Step | Detail |
|------|--------|
| Raw data | 21,605 rows |
| After cleaning | 20,684 rows |
| Train split (80%) | 16,547 samples |
| Test split (20%) | 4,137 samples |

**ขั้นตอน Preprocessing:**
1. กรองรีวิวที่สั้นเกินไป, มีค่าว่าง, หรือ sentiment ที่ไม่ถูกต้อง (`-1`, `0`, `1`)
2. สร้าง Input ในรูปแบบ `aspect <sep> sentence` เพื่อให้โมเดลรู้ว่ากำลังสนใจ Aspect ใด
3. แปลงข้อความด้วย `Tokenizer` + `pad_sequences` (maxlen=100)

**ตัวอย่างข้อมูล:**

| sentence | aspect | sentiment |
|----------|--------|-----------|
| I loved the story | plot | 1 |
| I found it extremely hard to keep reading it | storytelling | -1 |
| It lacked passion | plot | -1 |

---

## 🤖 Models

### Model 1: Bidirectional LSTM

**Architecture:** `Embedding → Bidirectional LSTM → Dense`

ใช้ **Class Weights** แก้ปัญหา Imbalanced Data:
- Neutral: `1.33x`
- Negative: `1.20x`

**Results:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.62 | 0.65 | 0.63 |
| Neutral | 0.52 | 0.48 | 0.50 |
| Positive | 0.77 | 0.77 | 0.77 |
| **Accuracy** | | | **0.67** |

---

### Model 2: Baseline — Logistic Regression

**Architecture:** `TF-IDF (ngram_range=(1,2)) → Logistic Regression`

**Results:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.67 | 0.62 | 0.65 |
| Neutral | 0.58 | 0.40 | 0.47 |
| Positive | 0.72 | 0.87 | 0.79 |
| **Accuracy** | | | **0.68** |

---

## 📈 Model Comparison

| Model | Overall Accuracy | Strength |
|-------|-----------------|----------|
| Baseline (Logistic Regression) | **68%** | Overall accuracy สูงกว่าเล็กน้อย |
| Bidirectional LSTM | 67% | ความเข้าใจเชิง semantic ดีกว่า |

---

## 🧪 Challenge Set: 4 สถานการณ์

### Case 1: Consistent Positive (+/+)
> *"The plot was amazing and the characters so perfect."*

| Model | Plot | Character |
|-------|------|-----------|
| LSTM | ✅ Positive (100.0%) | ✅ Positive (100.0%) |
| Baseline | ✅ Positive (75.0%) | ✅ Positive (75.2%) |

---

### Case 2: Consistent Negative (-/-)
> *"The movie was slow and the acting was terrible."*

| Model | Plot | Character |
|-------|------|-----------|
| LSTM | ✅ Negative (58.4%) | ✅ Negative (68.0%) |
| Baseline | ✅ Negative (40.0%) | ✅ Negative (44.8%) |

---

### Case 3: Consistent Neutral (0/0)
> *"Okay for a quick read and the characters so simple."*

| Model | Plot | Character |
|-------|------|-----------|
| LSTM | ✅ Neutral (76.6%) | ✅ Neutral (86.2%) |
| Baseline | ✅ Neutral (54.2%) | ✅ Neutral (55.6%) |

---

### Case 4: Conflict Case (+/-) ⚠️
> *"The plot was amazing, but the characters were boring."*

| Model | Plot (should be Pos) | Character (should be Neg) | Score |
|-------|----------------------|---------------------------|-------|
| LSTM | ✅ Positive (45.2%) | ❌ Positive (47.0%) | 50% |
| Baseline | ❌ Neutral (49.6%) | ❌ Neutral (45.4%) | **0%** |

**Analysis:** Baseline "สับสน" เพราะเห็นทั้งคำบวกและลบ จึงเฉลี่ยออกมาเป็น Neutral ทั้งหมด ส่วน LSTM ทำได้ดีกว่าโดยจับ Positive ของ plot ได้ถูก แต่ยังถูก Distract จากคำว่า "amazing" ที่อยู่ต้นประโยค

---

## 💡 Conclusion

| ด้าน | ผล |
|------|----|
| Overall Accuracy | Baseline ดีกว่าเล็กน้อย (68% vs 67%) |
| Semantic Understanding | **LSTM ดีกว่า** — เหมาะกับ ABSA มากกว่า |
| Conflict Case | LSTM ถูก 50%, Baseline ถูก 0% |
| Neutral Bias | Baseline มี bias ทาย Neutral เมื่อสับสน |
| Positive Bias | LSTM ถูก Distract จากคำ Positive ต้นประโยค |

**คำที่มีผลมากที่สุด:** `amazing`, `boring`, `but`

> LSTM มีความเข้าใจเชิงความหมายดีกว่า ซึ่งเหมาะกับงาน ABSA มากกว่า แม้ Overall Accuracy จะต่ำกว่า Baseline เพียง 1%

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras scikit-learn pandas numpy lime
```

### Run the Notebook

```bash
jupyter notebook "Aspect_Based_Sentiment_Analysis.ipynb"
```

### Load Pre-trained Models

```python
import tensorflow as tf
import joblib

# Load LSTM model
lstm_model = tf.keras.models.load_model('my_lstm_absa_model.h5')

# Load Baseline model
lr_model = joblib.load('my_baseline_lr_model.joblib')
```

---
