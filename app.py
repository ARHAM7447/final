from flask import Flask, render_template, request,  redirect
from flask_sqlalchemy import SQLAlchemy 
from flask_migrate import Migrate
from googletrans import Translator
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed for the first setup)
nltk.download('stopwords')

# Initialize the Flask app
app = Flask(__name__)

# Configure the app for the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'  # SQLite database file path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Secret key for sessions

# Set up the database and migrations
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize translator for multilingual support
translator = Translator()

# Prepare stopwords and the Porter Stemmer
stopwords_set = set(stopwords.words('english'))
porter = PorterStemmer()

# Define the Comment model
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    language = db.Column(db.String(10), nullable=False)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocessing function for comments
def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\W+', ' ', text.lower())  # Lowercase and remove special symbols
    words = text.split()

    # Handle negations
    for i in range(len(words)):
        if words[i] in ["not", "no", "never"] and i + 1 < len(words):
            words[i + 1] = f"NOT_{words[i + 1]}"

    words = [porter.stem(word) for word in words if word not in stopwords_set]
    return " ".join(words)

# Route to display the main sentiment analysis page
@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Detect the language and translate if necessary
        try:
            detected_language = translator.detect(comment).lang
            if detected_language != 'en':  # If not English, translate
                comment = translator.translate(comment, src=detected_language, dest='en').text
        except Exception:
            detected_language = "unknown"

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        # Save to the database
        new_comment = Comment(text=comment, sentiment=sentiment, language=detected_language)
        db.session.add(new_comment)
        db.session.commit()

        return render_template('index.html', sentiment=sentiment, language=detected_language)

    return render_template('index.html')

# Route to display history of comments
@app.route('/history', methods=['GET'])
def view_history():
    comments = Comment.query.all()  # Fetch all comments from the database
    return render_template('history.html', comments=comments)

# update and delete route 
@app.route('/update/<int:comment_id>', methods=['POST'])
def update_comment(comment_id):
    new_text = request.form.get('new_text')
    print(f"Received new_text: {new_text}")  # Debugging line
    comment = Comment.query.get_or_404(comment_id)
    if comment and new_text:  # Ensure new_text is not None
        comment.text = new_text
        db.session.commit()
    return redirect('/history')


@app.route('/delete/<int:comment_id>', methods=['POST'])
def delete_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)  # Fetch the comment from the database
    if comment:
        db.session.delete(comment)  # Delete the comment
        db.session.commit()  # Save the changes
        return redirect('/history')  # Redirect to the history page


# Main block to run the Flask application
if __name__ == '__main__':
    # Create the database if it doesn't exist
    with app.app_context():
        db.create_all()  # Create the tables for the Comment model
        print("Database created successfully!")
    app.run(debug=True)
