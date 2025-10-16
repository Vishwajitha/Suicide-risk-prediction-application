import torch
import numpy as np
from transformers import XLNetTokenizer, XLNetForSequenceClassification

# Define the correct path to your model
model_path = r"C:\Users\vishw\Downloads\goemotions_xlnet.pt"

# Define emotion labels (ensure this matches your training labels)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]  # Update this with your actual labels

# Check device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# Load XLNet model
num_labels = len(emotion_labels)
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)

# Load saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Function to predict emotions
def predict_emotions(text):
    encoding = tokenizer(
        text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Convert probabilities to emotion labels
    predicted_labels = [emotion_labels[i] for i, prob in enumerate(probs) if prob >= 0.5]
    return predicted_labels if predicted_labels else ["No strong emotion detected"]

# Directly passing a text variable
text_to_predict = "I’m so frustrated at everything. No one understands me, and I just want to disappear!"
predicted_emotions = predict_emotions(text_to_predict)

# Print results
print(f"Input Text: {text_to_predict}")
print(f"Predicted Emotions: {predicted_emotions}")
