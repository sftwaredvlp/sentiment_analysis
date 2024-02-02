import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns

final_data = pd.read_csv("C:/Users/hasan\Desktop/sentiment_analysis_content_recommendation/data_sentiment.csv")

def add_emojis(text):
    return f"📝 {text}"

def sentiment_analysis_page():
    st.title("Analysis Page 💬")

    # Add checkboxes with default values set to False
    show_content_1 = st.checkbox("Hashtag Histogram", value=False)
    show_content_2 = st.checkbox("Word Cloud", value=False)
    show_content_3 = st.checkbox("Sentiment Analysis Result", value=False)
    show_content_4 = st.checkbox("Sentiment Degrees with Hashtags", value=False)

    # Display content based on checkbox state
    if show_content_1:
        # Hashtag frequency 
        fig, ax = plt.subplots(figsize=(20, 10))
        final_data['hashtag'].value_counts().plot(kind='bar', color='orchid', ax=ax)
        plt.title('Hashtag Histogram')
        plt.xlabel('Hashtag')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    if show_content_2:
        # Word Cloud
        all_hashtags = ' '.join(final_data['hashtag'])
        wordcloud = WordCloud(width=800, height=400, background_color='black', 
                              colormap='viridis', collocations=False).generate(all_hashtags)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud')
        st.pyplot(fig)

    if show_content_3:
        # Sentiment Analysis Result
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=final_data, x='sentiment', hue='sentiment', multiple='stack', palette='viridis', shrink=0.8, ax=ax)
        ax.set_title('Sentiment Analysis Result')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    if show_content_4:
        # Sentiment Degrees with Hashtags
        fig, ax = plt.subplots(figsize=(50, 20))
        sns.barplot(data=final_data, x='hashtag', y='sentiment_score', palette='viridis', ax=ax)
        ax.set_title('Sentiment Degrees with Hashtags')
        ax.set_xlabel('Hashtag')
        ax.set_ylabel('Sentiment Degree')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.pyplot(fig)

def model_page():
    final_data["text"].fillna("", inplace=True)

    st.title("Sentiment Analysis and Hashtag Recommendation 💬🤖")

    user_input = st.text_input("Enter your tweet:")

    if user_input:
        entered_text_sentiment = TextBlob(user_input).sentiment.polarity
        entered_text_subjectivity = TextBlob(user_input).sentiment.subjectivity
        entered_text_sentiment_score = entered_text_sentiment * entered_text_subjectivity

        vectorizer = CountVectorizer()
        text_matrix = vectorizer.fit_transform(final_data["text"])

        entered_text_vectorized = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(entered_text_vectorized, text_matrix).flatten()
        combined_scores = similarity_scores + entered_text_sentiment_score * final_data["sentiment_score"]

        top_indices = combined_scores.argsort()[-3:][::-1]

        recommended_hashtags = final_data.loc[top_indices, "hashtag"].tolist()
        recommended_sentiment_scores = final_data.loc[top_indices, "sentiment_score"].tolist()

        unique_recommended_hashtags = list(set(recommended_hashtags))

        st.success(add_emojis(f"Entered Text: {user_input}"))
        st.success(add_emojis(f"Entered Text Sentiment Score: {entered_text_sentiment_score}"))
        st.success(add_emojis(f"Recommended Hashtags: {unique_recommended_hashtags}"))
        st.success(add_emojis(f"Recommended Sentiment Scores: {recommended_sentiment_scores}"))

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Select Page:", ["Analysis", "Model"])
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")

    st.sidebar.text("😀😁😂😃😄😅😆😆😇😈😉😀😁😂")
    st.sidebar.text("😊😋😌😍😎😏😐😑😒😓😔😀😁😂")
    st.sidebar.text("😕😖😗😘😙😚😛😜😝😞😟😀😁😂")
    st.sidebar.text("😠😡😢😣😤😥😦😧😨😩😪😀😁😂")
    st.sidebar.text("😫😬😭😮😮😯😰😱😲😳😴😀😁😂")
    st.sidebar.text("😋😌😍😎😏😐😑😒😓😔😕😀😁😂")
    
    if page == "Analysis":
        sentiment_analysis_page()
    elif page == "Model":
        model_page()

if __name__ == "__main__":
    main()
