
import os
import ssl
import nltk
import numpy as np
import pandas as pd
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Gradio kütüphanesini import edin
import gradio as gr

# SSL doğrulamasıyla ilgili bir sorun varsa kaldırmak için:
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK veri indirme
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("omw-1.4")
# Opsiyonel: Bazı durumlarda ihtiyaç duyulabilir
# nltk.download('punkt_tab')

###################################
# 1) Veri Yükleme
###################################

# CSV dosyasının aynı klasörde olduğunu varsayıyoruz.
dosya_yolu = "Hotel_Reviews.csv"  
data = pd.read_csv(dosya_yolu)

###################################
# 2) Veri Ön İşleme
###################################

# "United Kingdom" ifadesini "UK" ile değiştiriyoruz
data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom", "UK")
data.Hotel_Address = data.Hotel_Address.str.replace("Kingdom", "UK")

# Adresin son kelimesini ülke olarak çekiyoruz
data["countries"] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
data["countries"] = data["countries"].str.lower()

# İhtiyaç duyulmayan sütunları kaldırıyoruz
data.drop([
    'Additional_Number_of_Scoring', 
    'Review_Date', 
    'Reviewer_Nationality', 
    'Negative_Review', 
    'Review_Total_Negative_Word_Counts',
    'Total_Number_of_Reviews',
    'Positive_Review',
    'Review_Total_Positive_Word_Counts', 
    'Total_Number_of_Reviews_Reviewer_Has_Given',
    'Reviewer_Score', 
    'days_since_review', 
    'lat', 
    'lng'
], axis=1, inplace=True)

def impute(column):
    """Sütun içindeki string değerleri listeye (literal_eval) çeviriyoruz."""
    if isinstance(column, str):
        return literal_eval(column)
    else:
        return column

data['Tags'] = data['Tags'].apply(impute)
data['Tags'] = data['Tags'].str.lower()
data['Tags'] = data['Tags'].fillna("")
data['Tags'] = data['Tags'].astype(str)

###################################
# 3) Otel Öneri Fonksiyonu
###################################

def recommend_hotel(location: str, description: str):
    """
    Kullanıcının sağladığı konum ve açıklama (seyahat amacı) bilgisine göre
    otel önerir. Öneriler benzerlik skoru ve ortalama kullanıcı puanına göre 
    sıralanır.
    """
    # Metin ön işleme
    description = description.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Kullanıcının girdiği description'ı token'lara ayır, stopword ve noktalama temizle, lemmatize et
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(description)
        if word.isalnum() and word not in stop_words
    ]
    filtered_set = set(filtered_words)

    # Konuma göre filtreleme
    location = location.lower()
    country_hotels = data[data['countries'] == location].copy()
    country_hotels.reset_index(drop=True, inplace=True)

    # Benzerlik hesaplama
    similarities = []
    for tags in country_hotels['Tags']:
        tag_tokens = [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(tags.lower())
            if word.isalnum() and word not in stop_words
        ]
        tag_set = set(tag_tokens)
        similarity_score = len(filtered_set.intersection(tag_set))
        similarities.append(similarity_score)
    
    # Skorlara göre sıralama
    country_hotels['similarity'] = similarities
    country_hotels.sort_values(by=['similarity', 'Average_Score'], 
                               ascending=[False, False], 
                               inplace=True)
    # Aynı otelin birden çok kaydı varsa, ilkini tut
    country_hotels.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)

    # En çok 15 oteli göster
    return country_hotels[["Hotel_Name", "Average_Score", "Hotel_Address"]].head(15)

###################################
# 4) Gradio Arayüzü Fonksiyonu
###################################

def predict_hotels(location, description):
    """
    Gradio arayüzünden gelecek location ve description parametrelerini 
    kullanarak otelleri önerir.
    Çıktıyı metinsel bir tablo (Markdown) şeklinde geri döndürür.
    """
    df_results = recommend_hotel(location, description)
    return df_results.to_markdown(index=False)

###################################
# 5) Gradio Blocks Arayüzü
###################################

with gr.Blocks() as demo:
    gr.Markdown("## Otel Öneri Sistemi")
    gr.Markdown(
        "Bu sistem, kullanıcının seyahat amacı ve konumuna göre en uygun "
        "otel önerilerini sunar. Aşağıdaki kutucuklara bilgilerinizi girip "
        "‘Öneri Al’ butonuna basabilirsiniz."
    )

    with gr.Row():
        location_input = gr.Textbox(
            label="Konum (ülke)",
            placeholder="Örneğin: uk, france, spain vb."
        )
        description_input = gr.Textbox(
            label="Seyahat Açıklaması",
            placeholder="Örneğin: honeymoon, business trip, family vacation..."
        )

    output_box = gr.Textbox(
        label="Önerilen Oteller (Tablo)",
        lines=15
    )
    submit_btn = gr.Button("Öneri Al")

    submit_btn.click(fn=predict_hotels,
                     inputs=[location_input, description_input],
                     outputs=output_box)

# Lokal olarak `python app.py` derseniz çalışır;
# Hugging Face Spaces'te ise otomatik olarak aşağıdaki launch() tetiklenecektir.
if __name__ == "__main__":
    demo.launch()
