import streamlit as st
import pandas as pd
import numpy as np
import re

# ==========================================
# 1. CLASS TF-IDF MANUAL (Versi Compact + Transform)
# ==========================================
class TFIDF_Manual:
    def __init__(self):
        self.idf_dict = {}
        self.vocab = []
        self.word_to_idx = {}

    def fit_transform(self, data_series):
        # 1. Corpus
        corpus = data_series.apply(lambda x: str(x).split()).tolist()
        N = len(corpus)
        
        # 2. IDF Global
        doc_freq = {}
        for doc in corpus:
            for w in set(doc): doc_freq[w] = doc_freq.get(w, 0) + 1
        
        self.idf_dict = {w: np.log10(N / (count + 1)) for w, count in doc_freq.items()}
        self.vocab = sorted(self.idf_dict.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # 3. Matrix TF-IDF
        X_matrix = np.zeros((N, len(self.vocab)))
        for i, doc in enumerate(corpus):
            if not doc: continue
            counts = {}
            for w in doc: counts[w] = counts.get(w, 0) + 1
            for w, count in counts.items():
                if w in self.word_to_idx:
                    tf = count / len(doc)
                    X_matrix[i, self.word_to_idx[w]] = tf * self.idf_dict[w]

        # 4. Normalisasi
        norms = np.linalg.norm(X_matrix, axis=1, keepdims=True)
        return X_matrix / (norms + 1e-9)

    # --- METHOD TAMBAHAN KHUSUS GUI (Wajib Ada) ---
    def transform(self, text):
        words = str(text).split()
        vec = np.zeros(len(self.word_to_idx))
        if len(words) == 0: return vec
        
        counts = {}
        for w in words: counts[w] = counts.get(w, 0) + 1
        
        for w, count in counts.items():
            if w in self.word_to_idx:
                tf = count / len(words)
                vec[self.word_to_idx[w]] = tf * self.idf_dict.get(w, 0)
        
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

# ==========================================
# 2. CLASS KNN (Copy Paste Punya Kamu)
# ==========================================
class KNN:
    def __init__(self, k, weights):
        self.K = k; self.Weights = weights
    def fit(self, X, y):
        self.X_train = X; self.y_train = y
    def distance(self, x1, x2):
        return 1 - np.dot(x1, x2) # Cosine Distance
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    def _predict(self, x):
        dists = self.distance(self.X_train, x)
        best_k = np.argsort(dists)[:self.K]
        label_k = self.y_train[best_k]
        
        if self.Weights == 'uniform':
            vals, counts = np.unique(label_k, return_counts=True)
            return vals[np.argmax(counts)]
        elif self.Weights == 'distance':
            weights = 1 / (dists[best_k] + 1e-6)
            vote_scores = {}
            for lbl, w in zip(label_k, weights): vote_scores[lbl] = vote_scores.get(lbl, 0) + w
            return max(vote_scores, key=vote_scores.get)

# ==========================================
# 3. APLIKASI STREAMLIT
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource
def load_system():
    try: df = pd.read_csv('train.csv')
    except: return None, None
    
    # 1. Cleaning
    df['clean'] = df['sentence'].apply(clean_text)
    
    # 2. Training TF-IDF
    tfidf = TFIDF_Manual()
    X_train_vec = tfidf.fit_transform(df['clean']) # Return langsung X_matrix_norm
    
    # 3. Training KNN
    knn = KNN(k=15, weights='distance')
    knn.fit(X_train_vec, df['label'].values)
    
    return tfidf, knn

st.set_page_config(page_title="Meme Classifier", page_icon="🚫")
st.title("🚫 Meme Offense Prediction")
st.markdown("**Compact Code: TF-IDF Manual + KNN Cosine**")
st.divider()

tfidf_model, knn_model = load_system()

text_input = st.text_area("Masukkan Kalimat Meme:", height=100)

if st.button("Prediksi", type="primary"):
    if not text_input:
        st.warning("Isi dulu bos!")
    elif tfidf_model is None:
        st.error("File train.csv gak ketemu!")
    else:
        clean = clean_text(text_input)
        vec_input = tfidf_model.transform(clean) # Pakai method tambahan tadi
        pred = knn_model.predict([vec_input])[0]
        
        st.divider()
        if pred == 1:
            st.error("### ⚠️ OFENSIF")
            st.write("Kalimat ini terdeteksi kasar/menyinggung.")
        else:
            st.success("### ✅ AMAN")
            st.write("Kalimat ini terdeteksi netral.")