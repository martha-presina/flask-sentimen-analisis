from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    language = blob.detect_language()
    if language == 'en':
        score = blob.sentiment.polarity
        if score > 0:
            if score > 0.5:
                return "This sentence is highly positive."
            else:
                return "This sentence is positive."
        elif score == 0:
            return "This sentence is neutral."
        else:
            if score < -0.5:
                return "This sentence is highly negative." # -0.9
            else:
                return "This sentence is negative." # -0.3
    else:   
        an = blob.translate(from_lang='id', to='en')
        score = an.sentiment.polarity
        if score > 0:
            if score > 0.5:
                return "This sentence is highly positive."
            else:
                return "This sentence is positive."
        elif score == 0:
            return "This sentence is neutral."
        else:
            if score < -0.5:
                return "This sentence is highly negative." # -0.9
            else:
                return "This sentence is negative." # -0.3
