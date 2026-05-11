# Leukocyte Classification: Hybrid Analysis of Texture, Shape, and Color Features
### (Lökosit Sınıflandırması: Doku, Şekil ve Renk Özniteliklerinin Hibrit Analizi)

[English Summary below | Türkçe Özet aşağıdadır]

---

## 🌍 English Overview
This project focuses on the automated classification of leukocyte cells (Eosinophil, Neutrophil, Lymphocyte, and Monocyte) using hybrid feature fusion. 

- **Accuracy:** Achieved **93.95%** using Color Histograms and Random Forest. 
- **Key Techniques:** V6 Smart Crop, GLCM, LBP, HOG, and Wavelet Transform (Haar). 
- **Note:** The detailed technical report provided in this repository is in **Turkish**.

---

## 🇹🇷 Türkçe Özet
Bu çalışma, lökosit hücrelerinin doku, şekil, renk ve frekans tabanlı öznitelik yöntemleri kullanılarak otomatik sınıflandırılmasını içermektedir.
- **Veri Seti:** Kaggle Blood Cells veri seti (Eosinophil, Neutrophil, Lymphocyte, Monocyte).
- **Önemli Bulgular:** Eosinophil ve Neutrophil hücrelerinin ayırt edilmesinde renk özniteliklerinin kritik önemi kanıtlanmıştır.
- **Başarı Oranı:** En yüksek başarı %93.95 ile Renk Histogramı deneyinde elde edilmiştir. 

## 📊 Deney Sonuçları / Experimental Results
| Deney / Experiment | Yöntem / Method | Doğruluk / Accuracy |
| :--- | :--- | :--- |
| exp1_base_glcm | Sadece Doku (GLCM) | %76.85 |
| exp2_glcm_color | Doku + Renk | %87.40 |
| **exp_color_hist** | **Renk Histogramı** | **%93.95** |
| Bonus_wavelet | Wavelet (Boosted) | %78.00  |

## 📁 Dosyalar / Files
- `main.py`: Feature extraction and processing script.


*Sonuçlar, hücrelerin biyolojik boyanma özelliklerinden dolayı renk özniteliklerinin en ayırt edici faktör olduğunu kanıtlamıştır.*

## 📁 Dosya Yapısı
- `main.py`: Öznitelik çıkarımı ve ARFF dosyası oluşturma süreçlerini içeren ana kod.
