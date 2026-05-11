import os
import cv2
import numpy as np
import pywt # Wavelet
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# --- AYARLAR ---
DATASET_PATH = "dataset"
IMG_SIZE = (128, 128)

# --- DENEY KONFIGURASYONU ---
EXPERIMENTS = {
    "exp1_base_glcm.arff": ["glcm"],
    "exp2_glcm_lbp.arff":   ["glcm", "lbp"],
    "exp2_glcm_color.arff": ["glcm", "color"], 
    "exp3_glcm_lbp_hog.arff":   ["glcm", "lbp", "hog"],
    "exp3_glcm_lbp_color.arff": ["glcm", "lbp", "color"],
    "exp4_fusion_all.arff": ["glcm", "lbp", "hog", "hu", "color"],
    "exp_color_hist.arff": ["color_hist"]
}

def get_smart_crop(img):
    """ V6 Smart Crop Algoritması """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        h, w = img.shape[:2]
        return img[h//4:3*h//4, w//4:3*w//4]
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    if w < 20 or h < 20: 
        h_img, w_img = img.shape[:2]
        return img[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
    
    padding = 10
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - padding); y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding); y2 = min(h_img, y + h + padding)
    return img[y1:y2, x1:x2]

# --- ÖZELLİK ÇIKARIM FONKSİYONLARI ---

def extract_glcm(image_gray):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_gray, distances=[1], angles=angles, 
                        levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in props:
        feats.append(graycoprops(glcm, prop).mean())
    return feats

def extract_lbp(image_gray):
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.tolist()

def extract_hog(image_gray):
    features = hog(image_gray, orientations=8, pixels_per_cell=(32, 32),
                   cells_per_block=(1, 1), visualize=False)
    return features.tolist()

def extract_hu(image_gray):
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    return [-1 * np.sign(h) * np.log10(np.abs(h) + 1e-10) for h in hu]

def extract_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for i in range(3):
        feats.append(hsv[:,:,i].mean())
        feats.append(hsv[:,:,i].std())
    return feats

def extract_color_histogram(img):
    features = []
    for i in range(3): 
        hist = cv2.calcHist([img], [i], None, [8], [0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten().tolist())
    return features

# --- GÜNCELLENEN WAVELET FONKSİYONU ---
def extract_wavelet_all_bands(image_gray):
    """ 
    BONUS GÜNCELLEME: Sadece LL değil, HH, HL, LH bantlarını da kullanıyoruz.
    Böylece granülleri (detayları) yakalayıp skoru yükselteceğiz.
    Toplam 4 bant x 6 özellik = 24 Özellik dönecek.
    """
    # 'db1' (Haar) yerine 'db2' veya 'db4' biraz daha yumuşak ve iyi sonuç verebilir
    coeffs = pywt.dwt2(image_gray, 'db1') 
    LL, (LH, HL, HH) = coeffs
    
    all_features = []
    # 4 bandın her biri için GLCM çıkarıyoruz
    # HH bandı Eosinophil granüllerini yakalayacak!
    for band_name, band_img in zip(["LL", "LH", "HL", "HH"], [LL, LH, HL, HH]):
        # Görüntüyü 0-255 arasına normalize et (GLCM için şart)
        norm_img = cv2.normalize(band_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        all_features.extend(extract_glcm(norm_img))
        
    return all_features

# --- YARDIMCI FONKSİYONLAR ---

def write_header(f, filename, classes, active_feats, wavelet_mode=False):
    """ Dinamik Header Yazıcı """
    relation_name = filename.replace(".arff", "")
    f.write(f"@RELATION {relation_name}\n\n")
    
    if wavelet_mode:
        # Wavelet için özel başlık: 4 bant x 6 özellik
        bands = ["LL", "LH", "HL", "HH"]
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for b in bands:
            for p in props:
                f.write(f"@ATTRIBUTE {b}_{p} NUMERIC\n")
    else:
        # Diğer standart deneyler
        if "glcm" in active_feats:
            for n in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                f.write(f"@ATTRIBUTE glcm_{n} NUMERIC\n")
        if "lbp" in active_feats:
            for i in range(10): f.write(f"@ATTRIBUTE lbp_{i} NUMERIC\n")
        if "hog" in active_feats:
            for i in range(128): f.write(f"@ATTRIBUTE hog_{i} NUMERIC\n")
        if "hu" in active_feats:
            for i in range(7): f.write(f"@ATTRIBUTE hu_{i} NUMERIC\n")
        if "color" in active_feats: 
            for c in ['Hm','Hs','Sm','Ss','Vm','Vs']: f.write(f"@ATTRIBUTE color_{c} NUMERIC\n")
        if "color_hist" in active_feats: 
            for i in range(24): f.write(f"@ATTRIBUTE hist_val_{i} NUMERIC\n")
        
    f.write(f"@ATTRIBUTE class {{{','.join(classes)}}}\n\n")
    f.write("@DATA\n")

def main():
    print("Mühendislik Modu: Wavelet Skoru Yükseltiliyor...")
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    # 1. Dosyaları Aç
    file_handles = {}
    for filename, feats in EXPERIMENTS.items():
        f = open(filename, "w")
        write_header(f, filename, classes, feats, wavelet_mode=False)
        file_handles[filename] = f
        
    # Bonus Dosyası (Ayrı ve Özel Header)
    f_wavelet = open("bonus_wavelet_boosted.arff", "w")
    # Wavelet modunu True yapıyoruz ki 24 özelliği yazsın
    write_header(f_wavelet, "bonus_wavelet_boosted", classes, [], wavelet_mode=True)
    
    # 2. Görüntüleri İşle
    for label in classes:
        folder_path = os.path.join(DATASET_PATH, label)
        images = os.listdir(folder_path)[:500] 
        print(f"Sınıf işleniyor: {label}")
        
        for img_name in images:
            if not (img_name.lower().endswith(('.png', '.jpg', '.jpeg'))): continue
            
            # Okuma ve Kırpma
            path = os.path.join(folder_path, img_name)
            img = cv2.imread(path)
            if img is None: continue
            img = get_smart_crop(img)
            img = cv2.resize(img, IMG_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # --- ÖZELLİKLERİ BİR KERE HESAPLA ---
            features_pool = {
                "glcm": extract_glcm(gray),
                "lbp": extract_lbp(gray),
                "hog": extract_hog(gray),
                "hu": extract_hu(gray),
                "color": extract_color(img),
                "color_hist": extract_color_histogram(img)
            }
            
            # --- DOSYALARA DAĞIT ---
            for filename, required_feats in EXPERIMENTS.items():
                combined_vector = []
                for req in required_feats:
                    combined_vector += features_pool[req]
                line = ",".join(map(str, combined_vector)) + f",{label}\n"
                file_handles[filename].write(line)
            
            # --- BONUS WAVELET (Boosted) ---
            # Artık 4 banttan özellik çıkarıyoruz
            feat_wave = extract_wavelet_all_bands(gray)
            line_wave = ",".join(map(str, feat_wave)) + f",{label}\n"
            f_wavelet.write(line_wave)

    # 3. Kapat
    for f in file_handles.values(): f.close()
    f_wavelet.close()
    print("\n--- İŞLEM TAMAMLANDI ---")
    print("Oluşturulan dosya: 'bonus_wavelet_boosted.arff'")
    print("Bu dosya ile Weka'da çok daha yüksek bir skor bekliyoruz.")

if __name__ == "__main__":
    main()