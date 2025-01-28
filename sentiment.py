
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import urlparse, parse_qs
import time
import nltk

# **Download NLTK data**
nltk.download('vader_lexicon')

# **Initialize Sentiment Analyzer**
sia = SentimentIntensityAnalyzer()

# **YouTube API Configuration**
API_KEY = "AIzaSyCT4cESI9BbpLXn7PZXltewUKyczaAm-1o"  # Replace with your valid API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# **Function to Extract Video IDs**
def extract_video_ids(video_urls):
    """
    Extract video IDs from a list of YouTube video URLs.
    Args:
        video_urls (list): A list of YouTube video URLs.
    Returns:
        dict: A dictionary with video URLs as keys and their corresponding video IDs as values.
    """
    video_ids = {}
    for url in video_urls:
        parsed_url = urlparse(url.strip())
        video_id = None
        if parsed_url.hostname == "youtu.be":
            video_id = parsed_url.path[1:]  # Extract after "/"
        elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get("v", [None])[0]  # Extract "v" parameter
        video_ids[url] = video_id if video_id else None  # Mark invalid URLs with None
    return video_ids

# **Function to Fetch All Comments**
def fetch_all_comments(video_id):
    """
    Fetch all available comments from a YouTube video using the YouTube Data API.
    - video_id: ID of the YouTube video.
    Returns a list of comments.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        while request:
            response = request.execute()
            if 'items' not in response:
                break
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)  # Fetch next batch
    except Exception as e:
        st.error(f"Failed to fetch comments for video ID {video_id}: {e}")
    return comments

# **Function to Fetch Comments for Multiple Videos**
def fetch_comments_for_multiple_videos(video_ids):
    """
    Fetch comments for multiple videos sequentially to prevent rate limit issues.
    Args:
        video_ids (dict): A dictionary where keys are video URLs and values are video IDs.
    Returns:
        dict: A dictionary with video URLs as keys and their fetched comments as values.
    """
    all_comments = {}
    for url, video_id in video_ids.items():
        if video_id:
            st.write(f"Fetching comments for video: {url}")
            comments = fetch_all_comments(video_id)
            all_comments[url] = comments
            time.sleep(0.1)  # Delay between requests to avoid hitting rate limits
        else:
            st.warning(f"Invalid video URL: {url}")
    return all_comments

# **Function to Analyze Sentiments**
def analyze_sentiments(comments):
    """
    Analyze the sentiment of a list of comments.
    Returns a DataFrame with comments, sentiment labels, and sentiment scores.
    """
    sentiment_data = []
    for comment in comments:
        sentiment_score = sia.polarity_scores(comment)['compound']
        if sentiment_score > 0.05:
            sentiment_label = "Positive"
        elif sentiment_score < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        sentiment_data.append({
            "Comment": comment,
            "Sentiment": sentiment_label,
            "Score": sentiment_score
        })
    return pd.DataFrame(sentiment_data)

# **Streamlit Interface**
st.title("YouTube Multi-Video Sentiment Analysis")
st.subheader("Analyze comments from multiple YouTube videos simultaneously.")

# **Input multiple YouTube URLs**
video_urls_input = st.text_area("Enter YouTube Video URLs (one per line):", "")
video_urls = [url.strip() for url in video_urls_input.split("\n") if url.strip()]

# **Session State for Comments**
if "all_comments" not in st.session_state:
    st.session_state.all_comments = {}

# **Fetch Comments Button**
if st.button("Fetch Comments for All Videos"):
    if video_urls:
        # Extract video IDs
        video_ids = extract_video_ids(video_urls)
        invalid_urls = [url for url, video_id in video_ids.items() if not video_id]
        
        # Display invalid URLs
        if invalid_urls:
            st.warning(f"The following URLs are invalid or do not contain video IDs: {invalid_urls}")
        
        # Fetch comments
        st.info("Fetching comments for valid videos... This may take some time.")
        st.session_state.all_comments = fetch_comments_for_multiple_videos(video_ids)
        
        # Display results
        for url, comments in st.session_state.all_comments.items():
            if comments:
                st.success(f"Fetched {len(comments)} comments for video: {url}")
                st.write(f"Sample Comments for {url}:")
                st.write(comments[:5])  # Display first 5 comments
            else:
                st.warning(f"No comments found for video: {url}")
    else:
        st.error("Please provide at least one valid YouTube URL.")

# **Analyze Sentiments Button**
if st.button("Analyze Sentiments for All Videos"):
    if st.session_state.all_comments:
        st.info("Analyzing sentiments for all fetched comments...")
        for url, comments in st.session_state.all_comments.items():
            if comments:
                st.write(f"**Sentiment Analysis for Video:** {url}")
                sentiment_df = analyze_sentiments(comments)
                st.dataframe(sentiment_df)  # Display sentiment analysis DataFrame
                
                # **Display Sentiment Distribution**
                st.bar_chart(sentiment_df['Sentiment'].value_counts())
                
                # **Display Average Sentiment Score**
                avg_score = sentiment_df['Score'].mean()
                st.write(f"**Average Sentiment Score for {url}:** {avg_score:.2f}")

                # **Classify Overall Sentiment**
                if avg_score > 0.05:
                    overall_sentiment = "Positive 😊"
                elif avg_score < -0.05:
                    overall_sentiment = "Negative 😞"
                else:
                    overall_sentiment = "Neutral 😐"
                st.write(f"**Overall Sentiment for {url}:** {overall_sentiment}")
            else:
                st.warning(f"No comments available for analysis for video: {url}")
    else:
        st.warning("No comments available for analysis. Please fetch comments first.")