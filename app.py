from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample database of documents (you can replace this with your own data)
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a high-level programming language",
    "Flask is a lightweight WSGI web application framework",
    # Add more documents here...
]


# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template('index.html', error="Please enter some text.")

        # Transform the query into a TF-IDF vector
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarity between query and each document
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
        print("Similarity Scores:", similarity_scores) # Debugging statement

        # Threshold for plagiarism detection 
        threshold = 0.7

        # Identify similar documents
        similar_documents = [
            (documents[i], score) for i, score in enumerate(similarity_scores) if score > threshold
        ]
        print("Similar Documents:", similar_documents) # Debugging statement

        if not similar_documents:
            return render_template('index.html', error="No similar documents found.")

        return render_template('index.html', similar_documents=similar_documents, query=query)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
