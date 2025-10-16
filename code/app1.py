from flask import Flask, render_template, request, redirect, url_for, flash, session
import pyodbc
import torch
from transformers import BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification

app = Flask(__name__)
app.secret_key = 'vishwa25'  # Secret key for session management

# SQL Server connection string
connection_string = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-ISE16VE\\SQLEXPRESS;'
    'DATABASE=db_vnr;'
    'UID=sa;'
    'PWD=123456'
)

# Load the BERT model for suicide detection
model_path = r"C:\Users\vishw\Downloads\bert-suicide-detection"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load XLNet model for emotion classification
emotion_labels = ['hate', 'anger', 'love', 'worry', 'distress', 'happiness', 
                  'fun', 'empty', 'enthusiasm', 'sadness', 'surprise', 'boredom']
emotion_to_emoji = {
    'hate': '💢', 'anger': '😡', 'love': '❤️', 'worry': '😟', 
    'distress': '😖', 'happiness': '😊', 'fun': '😜', 'empty': '😶', 
    'enthusiasm': '🤩', 'sadness': '😢', 'surprise': '😲', 'boredom': '😒'
}

xlnet_model_path = r"C:\Users\vishw\Downloads\emotion_model.pt"
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=len(emotion_labels))

xlnet_model.load_state_dict(torch.load(xlnet_model_path, map_location=torch.device('cpu')))
xlnet_model.to("cpu")
xlnet_model.eval()

# Move BERT model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict suicide risk
def predict_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU if available

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return "Suicide risk" if predicted_class == 1 else "No suicide risk"

# Emotion prediction function
def predict_emotion(text):
    inputs = xlnet_tokenizer(text, padding="longest", truncation=True, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = xlnet_model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_emotion = emotion_labels[predicted_class]
    return f"{predicted_emotion} {emotion_to_emoji.get(predicted_emotion, '❓')}"

# Function to connect to SQL Server and check user credentials
def get_user_by_username(username):
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Users2 WHERE username=?", (username,))
        user = cursor.fetchone()  # Fetch the first matching user
        cursor.close()
        conn.close()
        return user
    except pyodbc.Error as e:
        print(f"Error: {e}")
        return None

# Function to register a new user
def register_user(username, password):
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Users2 (username, password) VALUES (?, ?)", (username, password))
        conn.commit()  # Commit the transaction
        cursor.close()
        conn.close()
    except pyodbc.Error as e:
        print(f"Error: {e}")

# Route to handle login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
       
        user = get_user_by_username(username)
        if user and user[2] == password:  # user[2] is the password column in the Users2 table
            session['user_id'] = user[0]  # Store user ID in session
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password!", "danger")
    
    return render_template("login.html")

# Route to handle signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if username already exists
        user = get_user_by_username(username)
        if user:
            flash("Username already exists!", "danger")
        else:
            register_user(username, password)
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for("login"))
    
    return render_template("signup.html")

# Flask route to handle tweet input
@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:  # Check if user is logged in
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))  # Redirect to login page

    if request.method == "POST":
        tweet_text = request.form["tweet_text"]
        prediction = predict_text(tweet_text)
        emotion = predict_emotion(tweet_text)

        result = {
            "tweet": tweet_text,
            "prediction": prediction,
            "emotion": emotion,
        }

        return render_template("index.html", result=result)

    return render_template("index.html")

# Logout route
@app.route("/logout")
def logout():
    session.pop('user_id', None)
    session.clear()
    flash("You have logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
