from helpers.utils import parse_whatsapp_chat
from flask import Flask, request, jsonify, render_template
from flask import request
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from datetime import datetime
import os

app = Flask(__name__)

# Ensure NLTK downloads are available
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Create "static" folder for generated images
if not os.path.exists("static"):
    os.makedirs("static")

# ✅ Define parse_whatsapp_chat BEFORE using it
    def parse_whatsapp_chat(file):
        content = file.read().decode("utf-8")
        messages = content.split("\n")

        data = []
        for msg in messages:
            parts = msg.split(" - ", 1)
            if len(parts) == 2:
                timestamp, message = parts
                user_msg = message.split(": ", 1)
                if len(user_msg) == 2:
                    user, msg_text = user_msg
                    data.append({"Timestamp": timestamp, "User": user, "Message": msg_text})

        return pd.DataFrame(data)


    import os
    from datetime import datetime
    import pandas as pd
    from flask import Flask, render_template, request, redirect, url_for
    import matplotlib.pyplot as plt

    app = Flask(__name__)

    # Set the upload folder for chat files
    UPLOAD_FOLDER = 'static/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    # Route for the home page (file upload form)
    @app.route('/')
    def index():
        return render_template('index.html')


    # Route to handle file uploads
    @app.route('/upload', methods=['POST'])
    def upload_file():
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('process_file', filename=file.filename))
        return 'Invalid file format. Please upload a .txt file.'


    # Route to process the uploaded file and generate daily message trends
    @app.route('/process/<filename>')
    def process_file(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Read the WhatsApp chat file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse the WhatsApp chat file
            data = []
            for line in lines:
                if line.startswith('['):
                    try:
                        date_str = line.split(']')[0][1:].strip()  # Extract date from the start of the line
                        message = line.split(']')[1].strip()  # Extract the message text
                        date = datetime.strptime(date_str, '%d/%m/%Y, %H:%M')  # Parse the date properly
                        data.append({'Date': date, 'Message': message})
                    except Exception as e:
                        print(f"Error parsing line: {line} -> {e}")  # Print out the error if something goes wrong
                        continue

            df = pd.DataFrame(data)
            print(df.head())  # Inspect the first few rows of the DataFrame

        # Create a pandas DataFrame
        df = pd.DataFrame(data)

        # Extract daily message counts
        df['Date'] = df['Date'].dt.date  # Extract just the date (without time)
        daily_trends = df.groupby('Date').size().reset_index(name='Message Count')

        # Generate the plot for daily trends
        fig, ax = plt.subplots()
        ax.plot(daily_trends['Date'], daily_trends['Message Count'], marker='o')
        ax.set(xlabel='Date', ylabel='Number of Messages', title='Daily Message Trends')

        # Save the plot as an image
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'daily_trends.png')
        plt.savefig(plot_path)
        plt.close()

        # Render the template to show the plot
        return render_template('trends.html', plot_path=plot_path)



    @app.route("/wordcloud", methods=["POST"])
    def generate_wordcloud():
        file = request.files.get("chat_file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = parse_whatsapp_chat(file)

        print(df.head())  # Debugging: Print first few rows
        print(df.columns)  # Debugging: Print column names

        if "Message" not in df.columns:
            return jsonify({"error": "No 'Message' column found"}), 400

        df = df[df["Message"].str.strip() != ""]  # Remove empty messages

        text = " ".join(df["Message"].dropna())

        if not text.strip():
            return jsonify({"error": "No valid text found for word cloud"}), 400

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        wordcloud.to_file("static/wordcloud.png")

        return jsonify({"message": "Word Cloud generated"})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/top_users", methods=["POST"])
def top_users():
    file = request.files.get("chat_file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = parse_whatsapp_chat(file)  # ✅ Fix
    user_counts = df["User"].value_counts().to_dict()

    return jsonify(user_counts)

@app.route("/wordcloud", methods=["POST"])
@app.route("/wordcloud", methods=["POST"])
def generate_wordcloud():
    file = request.files.get("chat_file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = parse_whatsapp_chat(file)  # ✅ Fix

    # Filter out empty messages
    df = df[df["Message"].str.strip() != ""]

    # Combine all messages into a single string
    text = " ".join(df["Message"].dropna())

    if not text.strip():  # If the text is still empty, return an error
        return jsonify({"error": "No valid text found for word cloud"}), 400

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file("static/wordcloud.png")

    return jsonify({"message": "Word Cloud generated"})


@app.route("/sentiment_analysis", methods=["POST"])
def sentiment_analysis():
    file = request.files.get("chat_file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = parse_whatsapp_chat(file)  # ✅ Fix
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(msg)["compound"] for msg in df["Message"].dropna()]

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"

    return jsonify({"sentiment": sentiment_label})
if __name__ == "__main__":
    app.run(debug=True)
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

app = Flask(__name__)

def process_daily_trends(df):
    # Ensure 'Date' column exists and convert it to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Group by date and count messages
    daily_counts = df.groupby(df['Date'].dt.date).size()

    # Plot daily trends
    plt.figure(figsize=(10, 5))
    daily_counts.plot(kind='line', marker='o', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.title('Daily Message Trends')
    plt.xticks(rotation=45)

    # Save plot as an image
    plt.savefig('static/daily_trends.png')
    plt.close()


@app.route('/message_trends', methods=['POST'])
def message_trends():
    file = request.files['file']
    if not file:
        return "Error: No file uploaded"

    df = pd.read_csv(file)

    # Generate the trends plot
    process_daily_trends(df)

    return render_template('index.html', trends_image='static/daily_trends.png')


if __name__ == "__main__":
    app.run(debug=True)
