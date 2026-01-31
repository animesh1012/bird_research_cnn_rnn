# 🐦 Bird Audio Data Preparation Research

This document explains the process and methodology followed for creating a high-quality, balanced dataset of bird audio recordings.

---

## 📍 Phase 1: Initial Data Collection

- Attempted to collect audio data for **42 bird species** from countries:
  ```
  ['Poland', 'Germany', 'Slovakia', 'Czech', 'Lithuania', 'Brazil', 'Spain']
  ```
- Some species had very limited recordings and some were not present in any of the target countries.
- After filtering based on availability, the dataset reduced to **28 species**.
- Applied a threshold requirement of **minimum 500 recordings per species**, resulting in only **23 species** being usable.

### **Result of Phase-1**
```
23 species × minimum 500 samples each
```

---

## 🌍 Phase 2: Expanded Data Collection

- Expanded collection efforts to **218 bird species** from **75 countries**.
- Out of these, only **145 species** had available recordings.
- After applying the **≥ 500 samples per species** rule, we retained **82 species**.

### **Result of Phase-2**
```
82 bird species having minimum 500 audio recordings
```

---

## ⚖️ Class Balancing & Chunking

- Each audio clip was **split into 15-second segments** for consistency.
- Performed **downsampling** to ensure all species had exactly **500 samples**.
- Prevents class imbalance and improves fairness in training.

---

## 📦 Final Dataset

- Each species (class) has **500 samples**
- Each sample length = **15-second audio**
- Total number of classes: **105**
- Total recordings: **105 × 500 = 52,500 audio files**

### **Final Dataset Summary**
| Property | Value |
|----------|-------|
| Countries covered | 75 |
| Raw collected species | 218 |
| Species retained (≥ 500 samples) | 105 |
| Final balanced samples | 52,500 |
| Clip duration | 15 seconds |

**🔗 Final Dataset Link**: https://drive.google.com/file/d/1dxPxKide39Nk_wQ72rkLKS_0u1OAOoxI/view?usp=sharing

---

## 🚀 Next Steps / Future Work
- Audio augmentation techniques for improving model generalization
- Multimodal learning combining audio + spectrograms
- Incorporation of metadata (location, season, call type)
- Model benchmarking across architectures (CNN, CRNN, Audio Transformers)

---

## 📝 Conclusion
After two major phases of data collection and balancing, we created a **high-quality, standardized, balanced dataset** of **52,500 samples**, ready for feature extraction and training in **bird species classification models**.

---
