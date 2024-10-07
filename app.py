from flask import Flask, request, render_template
import numpy as np
import math
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Loading the dataset and performing necessary preprocessing
def load_text_files(folder_path):
    """Reads all files in a folder and returns a dictionary
    with filenames as keys and content as values."""
    data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                data[filename] = file.read()

    return data

# Text cleaning function 
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits using regex
    text = re.sub(r'[^a-z\s]', '', text)  # Only keep letters and spaces
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return cleaned_tokens

path = './dataset' 
docs = load_text_files(path)

# Preprocess documents (cleaning and tokenization)
tokenized_docs = {filename: clean_text(doc) for filename, doc in docs.items()}

# Build vocabulary
vocab = set([word for doc in tokenized_docs.values() for word in doc])
vocab = sorted(vocab)

def term_frequency(term, document):
    return document.count(term) / len(document)

def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Calculate TF-IDF vectors for the documents
doc_tfidf_vectors = {filename: compute_tfidf(doc, tokenized_docs.values(), vocab) for filename, doc in tokenized_docs.items()}

# Route for the search query
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    tokenized_query = clean_text(query)  # Clean the query
    query_tfidf_vector = compute_tfidf(tokenized_query, tokenized_docs.values(), vocab)
    
    similarities = {filename: cosine_similarity(query_tfidf_vector, doc_vector) for filename, doc_vector in doc_tfidf_vectors.items()}
    
    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return render_template('results.html', query=query, results=ranked_docs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
