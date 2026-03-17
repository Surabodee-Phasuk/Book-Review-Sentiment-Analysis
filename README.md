# Book-Review-Sentiment-Analysis
Understanding Reader Emotions through Aspect-Based Sentiment Analysis with by team 3 people

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)](https://scikit-learn.org/)

โครงการวิเคราะห์ความรู้สึกจากรีวิวหนังสือแบบระบุแง่มุม (**Aspect-Based Sentiment Analysis**) โดยเปรียบเทียบประสิทธิภาพระหว่างโมเดล Machine Learning พื้นฐาน และ Deep Learning เพื่อการวิเคราะห์ที่แม่นยำในระดับองค์ประกอบของเนื้อหา

## 🌟 บทนำ (Introduction)
โดยปกติการทำ Sentiment Analysis จะบอกภาพรวมว่ารีวิวเป็นบวกหรือลบ แต่โปรเจกต์นี้เลือกใช้ **ABSA** เพื่อเจาะลึกลงไปว่าผู้อ่านรู้สึกอย่างไรต่อแง่มุมเฉพาะ เช่น:
- **Plot**: เนื้อเรื่องและการดำเนินเรื่อง
- **Character**: ตัวละครและการพัฒนาตัวละคร
- **Storytelling**: วิธีการเล่าเรื่องของผู้เขียน
- **Price**: ความคุ้มค่าและราคา

## 📊 ข้อมูลที่ใช้ (Dataset)
- **จำนวนข้อมูล**: 21,605 แถว (ทำความสะอาดแล้วเหลือ 20,684 แถว)
- **รูปแบบ Input**: ใช้เทคนิคการรวม Aspect เข้ากับประโยคในรูปแบบ `aspect <sep> sentence`
- **การแบ่งข้อมูล**: Train 80% / Test 20%

## 🤖 โมเดลที่ใช้เปรียบเทียบ (Models)
1. **Baseline Model**: Logistic Regression ร่วมกับ TF-IDF Vectorizer
2. **Deep Learning Model**: Long Short-Term Memory (LSTM)

## 📈 สรุปผลการทดลอง
จากการทดสอบพบว่าโมเดล **LSTM** มีความสามารถในการเข้าใจบริบท (Context) ได้ดีกว่า โดยเฉพาะในประโยคที่มีความย้อนแย้ง (เช่น "เนื้อเรื่องดีแต่ตัวละครน่าเบื่อ") ในขณะที่ Baseline Model มักจะเกิดความสับสนเมื่อเจอคำที่มีความหมายต่างกันในประโยคเดียว

## 📂 โครงสร้างไฟล์ (Project Structure)
| ไฟล์ | รายละเอียด |
|---|---|
| `Aspect_Based_Sentiment_Analysis.ipynb` | สมุดโน้ตขั้นตอนการเตรียมข้อมูลและการเทรนโมเดล |
| `preprocessed_review.csv` | ชุดข้อมูลที่ผ่านการ Pre-process เรียบร้อยแล้ว |
| `my_lstm_absa_model.h5` | โมเดล LSTM ที่บันทึกไว้ |
| `my_baseline_lr_model.joblib` | โมเดล Logistic Regression ที่บันทึกไว้ |
| `Cs374_NLP_Book_Final.pdf` | เอกสารสรุปโครงการและผลการวิจัยฉบับเต็ม |

## 🚀 การเริ่มต้นใช้งาน
1. **ติดตั้ง Dependencies**:
   ```bash
   pip install tensorflow scikit-learn pandas numpy joblib
