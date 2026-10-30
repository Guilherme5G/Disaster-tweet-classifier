from sklearn.feature_extraction.text import TfidfVectorizer
documents = [
    "fire near town",
    "fire near school",
    "music near school"
]

vectorizer = TfidfVectorizer(norm=None) # o sklearn ja normaliza, porem vamos tirar essa normalizacao para fazer o calculo de tdidf na mao

tfidf_matrix = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray().round(3))
print(tfidf_matrix.shape)

