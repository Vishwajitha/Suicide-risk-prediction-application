from flask import Flask, render_template, request, redirect, url_for, flash, session
import pyodbc
from transformers import BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification
import torch
import pandas as pd
import tweepy
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
emotion_labels = ['hate',
 'anger',
 'love',
 'worry',
 'distress',
 'happiness',
 'fun',
 'empty',
 'enthusiasm',
 'sadness',
 'surprise',
 'boredom']
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

@app.route("/about")
def about():
    return render_template("aboutus.html")

@app.route("/check_file_type", methods=["POST"])
def check_file_type():
    file = request.files.get("file")
    if file:
        ext = file.filename.lower().split('.')[-1]
        if ext == 'jpg':
            flash("Invalid file type! Please upload a CSV or Excel file.", "danger")
        else:
            flash("File ready for prediction.", "success")
    return redirect(url_for("index"))


# Route to handle file upload and predictions
@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return redirect(url_for("login"))
    
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("No file uploaded!", "danger")
            return redirect(url_for("index"))

        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(file)
            else:
                flash("Invalid file format! Please upload a CSV or Excel file.", "danger")
                return redirect(url_for("index"))

            if 'Tweet Text' in data.columns and 'Author Username' in data.columns:
                data['Prediction'] = data['Tweet Text'].apply(predict_text)
                data['Emotion'] = data['Tweet Text'].apply(predict_emotion)
                result = data[['Author Username', 'Tweet Text', 'Prediction', 'Emotion']].values.tolist()
                # Store the result in the session
                session['result'] = result  # Store result in session
            else:
                flash("File must contain 'Tweet Text' and 'Author Username' columns.", "danger")
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")

    return render_template("index.html", result=result)

   
# Route to display results in a separate page
@app.route("/results", methods=["GET"])
def results():
    if 'user_id' not in session:
        return redirect(url_for("login"))

    # You should already have 'result' data from previous processing in the session
    # For now, let's just assume 'result' is stored in the session
    result = session.get('result', None)  # Retrieve the result from the session
    
    if result:
        # Count Suicide Risk and No Suicide Risk
        suicide_risk_count = sum(1 for r in result if r[2] == 'Suicide risk')  # Index 2 is the Prediction column
        no_suicide_risk_count = sum(1 for r in result if r[2] == 'No suicide risk')

        # Prepare visualization (optional, you can use libraries like Matplotlib or Plotly)
        # Example: Pie chart visualization
        import matplotlib.pyplot as plt
        import io
        import base64
        from collections import Counter    
        fig, ax = plt.subplots()
        ax.pie([suicide_risk_count, no_suicide_risk_count], labels=['Suicide Risk', 'No Suicide Risk'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

        # Save the plot to a BytesIO object
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)

        # Convert the plot to a base64 string so it can be embedded in the HTML
        plot_url = base64.b64encode(img_io.getvalue()).decode('utf8')
        # Count Emotion Occurrences
        emotion_counts = Counter([r[3].split(" ")[0] for r in result])  # Extracting emotion label from "emotion emoji"

        # Create Bar Chart for Emotions
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Count")
        ax.set_title("Emotion Analysis")
        ax.set_xticklabels(emotion_counts.keys(), rotation=45)

        img_io = io.BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)
        bar_chart_url = base64.b64encode(img_io.getvalue()).decode('utf8')

        return render_template('results.html', result=result, suicide_risk_count=suicide_risk_count, no_suicide_risk_count=no_suicide_risk_count, plot_url=plot_url,bar_chart_url=bar_chart_url)
    else:
        flash("No results available", "danger")
        return redirect(url_for("index"))
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


# Function to retrieve decrypted configuration values from the database
def get_config_value(key):
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        # Call the stored procedure to get the decrypted value
        cursor.execute("EXEC sp_GetDecryptedConfig @key = ?", key)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[1] if result else None  # result[1] because SELECT returns config_key and config_value
    except pyodbc.Error as e:
        print(f"Error retrieving {key}: {e}")
        return None


@app.route("/send_messages", methods=["POST"])
def send_messages():
    if 'user_id' not in session:
        return redirect(url_for("login"))

    result = session.get('result', None)  # Get stored results
    if not result:
        flash("No results available to send messages!", "danger")
        return redirect(url_for("index"))
    
    # Retrieve API credentials from the database
    api_key = get_config_value('api_key')
    api_secret = get_config_value('api_secret')
    bearer_token = get_config_value('bearer_token')
    access_token = get_config_value('access_token')
    access_token_secret = get_config_value('access_token_secret')

    if not all([api_key, api_secret, bearer_token, access_token, access_token_secret]):
        flash("Missing API credentials in the database!", "danger")
        return redirect(url_for("index"))

    # Set up Twitter API authentication
    client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_token_secret)
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)

    # Retrieve the predefined message from the database
    suicide_message = get_config_value('suicide_message')
    if not suicide_message:
        flash("No predefined message found in the database!", "danger")
        return redirect(url_for("index"))

    # Filter users at suicide risk
    suicide_risk_users = [row for row in result if row[2] == "Suicide risk"]  # Index 2 is 'Prediction' column

    if not suicide_risk_users:
        flash("No users detected at suicide risk.", "info")
        return redirect(url_for("index"))

    # Send messages
    for row in suicide_risk_users:
        username = row[0]  # 'Author Username' is at index 0
        tweet_text = (
        f"@{username} {suicide_message} "
         f"Here's help: https://connectingngo.org/#:~:text=Connecting%20Trust%20is%20a%20Pune,have%20lost%20someone%20to%20suicide."
    )
        try:
            client.create_tweet(text=tweet_text)
            flash(f"Message sent to @{username}", "success")
        except Exception as e:
            flash(f"Failed to send message to @{username}: {str(e)}", "danger")

    # Clear the session and log out
    session.clear()
    flash("Messages sent successfully! You have been logged out.", "success")
    
    return redirect(url_for("login"))  # Redirect to login page

# Logout route
@app.route("/logout", methods=["GET", "POST"])
def logout():
    if request.method == "POST":
        session.pop('user_id', None)
        session.clear()
        flash("You have logged out.", "info")
        return redirect(url_for("login"))
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)