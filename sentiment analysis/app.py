import streamlit as st
from huggingface_sentiment import setup_sentiment_analyzer, analyze_text

# Initialize the sentiment analyzer
analyzer = setup_sentiment_analyzer()

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🤖")
    
    st.title("Sentiment Analysis")
    
    text = st.text_area("Enter text to analyze:", placeholder="Type your text here...", height=150)
    
    if st.button("Analyze Sentiment"):
        if text.strip():
            result = analyze_text(analyzer, text)
            
            if result['sentiment'] == 'Positive':
                st.success(f"✅ Positive ({result['confidence']:.1f}% confidence)")
            else:
                st.error(f"❌ Negative ({result['confidence']:.1f}% confidence)")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()