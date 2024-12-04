# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:15:15 2024

@author: José
"""
import PyPDF2
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append(r'D:\python_utils')
import graphers

# Ensure nltk resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Define Portuguese stopwords and stemmer
stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer("portuguese")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and stem words
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Keyword frequency tracker
def track_keywords(tokens, keywords):
    keyword_counts = Counter(tokens)
    kw_tokens = set([preprocess_text(word)[0] for word in keywords])
    tracked_counts = sum([keyword_counts.get(keyword, 0) for keyword in kw_tokens])
    tracked_counts_freq = 10000*tracked_counts / len(tokens)
    return tracked_counts_freq


# Define keywords of interest
# Track keyword frequencies
kwds_vals = [
    'gênero', 'mulher', 'feminismo', 'feminista', 'transsexual',
    # gênero
    'machismo', 'sexualidade', 'lgbt', 'lgbtq', 'lgbtqia', 'lgbtq+',
    'homofobia', 'casamento', # orientação sexual
    'racismo', 'racialismo',
    'negritude', 'PPI', 'antirracismo', 'cotas', # racismo
    'minoria', 'discriminação',
    'diversidade', 'inclusão', 'acessível' # genéricas
    ]


kwds_trabalho = [
    'trabalho', 'emprego', 'salário', 'mínimo', 'renda', 'sindicato'
    ]

kwds_desenvol = [
    'inflação', 'crescimento', 'desenvolvimento', 'indutor', 'economia', 'PIB',
    'infraestrutura', 'PAC', 'indústria', 
    ]

kwds_anticapital = [
    'fortunas', 'dívida', 'auditoria', 'capital', 'câmbio', 'rentismo',
    'rentista', 'dólar', 'FMI', 'estrangeiro', 'juros',
    'bancos', 'agronegócio'
    ]

kwds_assist = [
    'distribuição', 'previdência', 'aposentadoria', 'redistribuição', 'pobreza',
    'miséria', 'moradia', 'bolsa', 'cuidar', 'casa', 'MCMV', 'terra', 'agrária',
    'camponeses', 'camponês', 'agricultor'
    ]

kwds_democracia = [
    'democracia', 'fascismo', 'golpe', 'impeachment', 'golpista',
    'ditadura', 'autoritarismo', 'autoritário'
    ]

kwds_lula = [
    'lindu', 'metalúrgico', 'retirante', 'marisa', 'lula'
    ]


# Function to prepare data for LDA
def prepare_corpus(texts):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(stop_words))
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer

# Function to apply LDA and extract topics
def extract_topics(dtm, vectorizer, n_topics=5, n_top_words=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {topic_idx+1}"] = top_words
    
    return lda, topics

# Example usage
pdf_paths = os.listdir(r'D:\Economia\Politica Brasil\Its the Culture, Stupid\Documentos\PT')

# Análise contanto textos
dict_result = {}
for pdf in pdf_paths:
    path = f'D:\\Economia\\Politica Brasil\\Its the Culture, Stupid\\Documentos\\PT\\{pdf}'
    text = extract_text_from_pdf(path)
    tokens = preprocess_text(text)
    vals_freq = track_keywords(tokens, kwds_vals)
    desenvol_freq = track_keywords(tokens, kwds_desenvol)
    anticapital_freq = track_keywords(tokens, kwds_anticapital)

    trabalho_freq = track_keywords(tokens, kwds_trabalho)
    assist_freq = track_keywords(tokens, kwds_assist)
    democracia_freq = track_keywords(tokens, kwds_democracia)
    lula_freq = track_keywords(tokens, kwds_lula)
    
    ano = (pdf[:4])

    dict_result[ano] = [
        trabalho_freq, desenvol_freq, anticapital_freq,
        assist_freq, vals_freq, democracia_freq,
        lula_freq
        ]

df_result = pd.DataFrame(
    dict_result,
    index=['trabalho', 'desenvolvimento', 'anticapital', 'assistencialismo',
           'valores', 'democracia', 'Lula']).T

graf = graphers.gen_graf(df_result, 'Frequência Relativa de Termos em Programas de Governo do PT')
graf.write_image(r'D:\Economia\Politica Brasil\Its the Culture, Stupid\teste.png')




















