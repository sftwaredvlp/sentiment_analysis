In this project, data was collected from Twitter, focusing on text content. The data cleaning process involved removing emojis, stop words, and non-English words. Additionally, duplicate texts were identified and removed to ensure data quality.

The sentiment analysis was performed using the TextBlob library. Each text in the dataset was iterated through, and sentiment labels (Positive, Negative, Neutral) were assigned based on polarity scores. The results were then added to the dataset as "sentiment" and "sentiment_score" columns.

Handling missing values in the "text" column was addressed by filling empty values with an empty string, ensuring compatibility with the CountVectorizer.

For the model creation phase:

    Entered Text for Prediction:
        A mechanism to define the entered text for which similar hashtags were to be found.

    Sentiment Analysis for Entered Text:
        Utilized TextBlob to calculate the sentiment score for the entered text.

    CountVectorizer:
        Created a CountVectorizer instance to convert text data into a matrix of token counts.

    Cosine Similarity Calculation:
        Transformed text data using CountVectorizer and calculated cosine similarity between the entered text and each row in the text matrix.

    Combining Similarity and Sentiment Scores:
        Combined the similarity scores with the sentiment score of the entered text for ranking.

    Top Three Similar Texts:
        Determined the indices of the top three most similar texts.

    Retrieve Hashtags and Sentiment Scores:
        Retrieved corresponding hashtags and sentiment scores for the top three similar texts.

    Remove Duplicate Hashtags:
        Removed duplicates from recommended hashtags.

The final stage involved visualization, where hashtag histogram, word cloud, sentiment analysis results, and sentiment degrees with hashtags were generated.

Finally, the project was deployed using Streamlit, featuring an analysis and model page. Users can input a tweet, and the system provides three recommended hashtags.

Project Link: https://sentimentanalysis-content-recommendation.streamlit.app/ 
