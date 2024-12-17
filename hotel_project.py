
import nltk
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("wordnet")
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')     
nltk.download('punkt')         
nltk.download('omw-1.4')       
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

dosya_yolu = "/Users/miray/Downloads/Hotel_Reviews.csv"
data = pd.read_csv(dosya_yolu)

data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom", "UK")
data.Hotel_Address = data.Hotel_Address.str.replace("Kingdom", "UK")

data["countries"] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
print(data.countries.unique())

data.drop(['Additional_Number_of_Scoring', 'Review_Date', 'Reviewer_Nationality', 'Negative_Review', 'Review_Total_Negative_Word_Counts',
           'Total_Number_of_Reviews','Positive_Review','Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given',
           'Reviewer_Score', 'days_since_review', 'lat', 'lng'], axis=1, inplace=True)

def impute(column):
    if isinstance(column, str):  
        return literal_eval(column)  
    else:
        return column 
    
data['Tags'] = data['Tags'].apply(impute)
data['countries'] = data['countries'].str.lower()
data['Tags'] = data['Tags'].str.lower()
print("Eksik veri sayısı:", data['Tags'].isnull().sum())
data['Tags'] = data['Tags'].fillna("")
data['Tags'] = data['Tags'].astype(str)

nltk.download('punkt_tab')

def recommend_hotel(location, description):
    """
    Kullanıcının sağladığı konum ve açıklamaya göre otel önerir.
    Önerilen oteller benzerlik skoruna ve ortalama kullanıcı derecelendirmesine göre sıralanır.
    """
    description = description.lower()  
    stop_words = set(stopwords.words('english'))  
    lemmatizer = WordNetLemmatizer() 
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(description)
        if word.isalnum() and word not in stop_words
    ]
    filtered_set = set(filtered_words)  

    location = location.lower()
    country_hotels = data[data['countries'] == location].copy()
    country_hotels.reset_index(drop=True, inplace=True)

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
    country_hotels['similarity'] = similarities
    country_hotels = country_hotels.sort_values(by=['similarity', 'Average_Score'], 
                                                ascending=[False, False])
    country_hotels = country_hotels.drop_duplicates(subset='Hotel_Name', keep='first')
    return country_hotels[["Hotel_Name", "Average_Score", "Hotel_Address"]].head(15)

print(recommend_hotel('UK','I am going on a honeymoon, I need a honeymoon suite room for 3 nights'))
