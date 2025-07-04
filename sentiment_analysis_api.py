import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

app = FastAPI(
    title="Sentiment Analysis API",
    description="Uses VADER for sentiment analysis."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        if not isinstance(text, str):  
            return "" 
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(text)
        lm_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lm_tokens)
    except Exception as e:
        print(f"Error occurred: {e}")
        return ""

def analyze_sentiment(text):
    if isinstance(text, str):
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        sentiment_label = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"
        return sentiment_label, compound_score
    return "Neutral", 0

def calculate_sentiment_percentages(df):
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    sentiment_percentages = {
        "Positive": sentiment_counts.get("Positive", 0),
        "Negative": sentiment_counts.get("Negative", 0),
        "Neutral": sentiment_counts.get("Neutral", 0)
    }
    return sentiment_percentages

def highlight_sentiment_words(text, sentiment_label):
    words = text.split()
    highlighted_words = []
    
    for word in words:
        sentiment_score = analyzer.polarity_scores(word)['compound']
        
        if sentiment_label == "Positive" and sentiment_score >= 0.05:
            highlighted_words.append(f"<span class='positive-word'>{word}</span>")  
        elif sentiment_label == "Negative" and sentiment_score <= -0.05:
            highlighted_words.append(f"<span class='negative-word'>{word}</span>")  
        else:
            highlighted_words.append(word)  
    
    return ' '.join(highlighted_words)

@app.get("/health")
def health_check():
    return {"status": "API is running!"}

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        processed_text = preprocess_text(input_data.text)
        sentiment_label, score = analyze_sentiment(processed_text)
        highlighted_text = highlight_sentiment_words(processed_text, sentiment_label)
        
        return JSONResponse(content={
            "highlighted_text": highlighted_text,
            "sentiment": sentiment_label,
            "score": score
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), output_path: str = None):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents), encoding="ISO-8859-1")
        
        if 'reviewText' not in df.columns:
            raise ValueError("CSV file must contain 'reviewText' column.")
        
        df['Processed_Text'] = df['reviewText'].apply(preprocess_text)
        df[['Sentiment', 'Score']] = df['Processed_Text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
        df.dropna(inplace=True)
        
        sentiment_percentages = calculate_sentiment_percentages(df)

        if output_path:
            df.to_csv(output_path, index=False)
            return JSONResponse(content={"message": f"Predictions saved to: {output_path}"})

        return JSONResponse(content={
            "message": "Predictions completed!",
            "sentiment_percentages": sentiment_percentages,
            "predictions": df.to_dict(orient="records")
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
