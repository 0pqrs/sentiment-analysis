# Sentiment Analysis using real Hugging Face models
# This code uses the Transformers library to analyze sentiment in text

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def setup_sentiment_analyzer():
    """Set up the sentiment analysis pipeline using a pre-trained model."""
    print("Loading sentiment analysis model from Hugging Face...")
    
    # Use a popular sentiment analysis model from Hugging Face
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    try:
        # Create a sentiment analysis pipeline directly
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        print(f"Successfully loaded model: {model_name}")
        return sentiment_analyzer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Alternative approach if pipeline fails
        print("Trying alternative approach...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            def manual_sentiment(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.detach().numpy()[0]
                
                if scores[1] > scores[0]:  # Positive has higher score
                    return [{"label": "POSITIVE", "score": float(scores[1])}]
                else:
                    return [{"label": "NEGATIVE", "score": float(scores[0])}]
            
            print("Successfully created manual sentiment analyzer")
            return manual_sentiment
        
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            raise

def analyze_text(analyzer, text):
    """Analyze the sentiment of the provided text."""
    try:
        result = analyzer(text)
        
        if isinstance(result, list):
            result = result[0]  # Get first result
            
        label = result["label"]
        score = result["score"]
        
        # Format the result
        sentiment = "Positive" if label == "POSITIVE" else "Negative"
        confidence = round(score * 100, 2)
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_result": result
        }
    
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {
            "text": text,
            "error": str(e)
        }

def analyze_batch(analyzer, texts):
    """Analyze sentiment for a batch of texts."""
    results = []
    for text in texts:
        results.append(analyze_text(analyzer, text))
    return results

def interactive_mode(analyzer):
    """Run the sentiment analyzer in interactive mode."""
    print("\n==== Hugging Face Sentiment Analysis ====")
    print("Type 'exit', 'quit', or 'q' to end the program.")
    
    while True:
        print("\nEnter text to analyze:")
        text = input("> ")
        
        if text.lower() in ["exit", "quit", "q"]:
            print("Exiting program.")
            break
        
        if not text.strip():
            print("Please enter some text to analyze.")
            continue
        
        result = analyze_text(analyzer, text)
        
        print("\nAnalysis Result:")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")

def main():
    print("Initializing Hugging Face Sentiment Analysis")
    
    try:
        # Set up the sentiment analyzer
        analyzer = setup_sentiment_analyzer()
        
        # Sample examples to demonstrate functionality
        examples = [
            "I absolutely love this product! It's amazing and works perfectly.",
            "This is terrible. I'm very disappointed with the quality.",
            "It's okay. Nothing special but it gets the job done."
        ]
        
        print("\n--- Testing with sample texts ---")
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}: \"{example}\"")
            result = analyze_text(analyzer, example)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}%")
        
        # Start interactive mode
        interactive_mode(analyzer)
    
    except Exception as e:
        print(f"Error in main program: {e}")

if __name__ == "__main__":
    main()