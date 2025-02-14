from django.shortcuts import render
from scraper.mpr import model,get_gemini_response
from scraper.charts import preprocess_data,analyze_sentiment,plot_top_negative_reviews,plot_top_positive_reviews
import pandas as pd
import json
def home(request):
    return render(request,'main.html')


def recommend_product(df):
    # Calculate sentiment scores
    sentiment_counts = df['sentiment'].value_counts()
    total_reviews = sentiment_counts.sum()
    positive_reviews = sentiment_counts.get('positive', 0)
    negative_reviews = sentiment_counts.get('negative', 0)
    
    # Calculate sentiment ratio
    if total_reviews > 0:
        sentiment_ratio = positive_reviews / total_reviews
    else:
        sentiment_ratio = 0.5  # Default to neutral if no reviews
    
    # Define recommendation logic
    if sentiment_ratio >= 0.6:
        recommendation = "Based on user reviews and our score, should consider buying this product."
    elif sentiment_ratio <= 0.4:
        recommendation = "Based on user reviews and our score, you might want to reconsider buying this product."
    else:
        recommendation = "Based on user reviews and our score, the sentiment towards this product is mixed."
    
    return recommendation



def review_analytics_view(request):
    url = request.POST.get("producturl")

    gemini_result = get_gemini_response(url)
    product_name = gemini_result["product_name"]
   
    # Read the CSV file and preprocess the data
    df = pd.read_csv(f"{product_name}.csv")
    df = preprocess_data(df)
    
    # Generate data for star rating distribution chart
    star_rating_data = df['star_rating'].value_counts().sort_index().to_dict()
    star_ratings = list(star_rating_data.keys())
    star_counts = list(star_rating_data.values())
    
    # Generate data for reviews per date chart
    reviews_per_date = df.groupby('date').size().reset_index(name='Number of Reviews')
    review_dates = reviews_per_date['date'].tolist()
    review_counts = reviews_per_date['Number of Reviews'].tolist()

    most_busy_month_data = df['Month'].value_counts().sort_index().to_dict()
    most_busy_month_labels = list(most_busy_month_data.keys())
    most_busy_month_values = list(most_busy_month_data.values())
    
    # Generate data for sentiment analysis chart
    sentiment_data = analyze_sentiment(df)
    
    monthly_avg_rating = df.groupby(['Year', 'Month'])['star_rating'].mean().reset_index()
    monthly_avg_rating['Month'] = monthly_avg_rating['Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
    monthly_avg_rating_labels = monthly_avg_rating.apply(lambda x: f"{x['Month']} {int(x['Year'])}", axis=1).tolist()
    monthly_avg_rating_values = monthly_avg_rating['star_rating'].tolist()
    
    top_positive_reviews_data = plot_top_positive_reviews(df,max_words=10)
    top_negative_reviews_data = plot_top_negative_reviews(df,max_words=10)

    recommendation = recommend_product(df)
    gemini_response=gemini_result["formatted_response"]
    # Pass data to the template
    context = {
        'star_ratings': json.dumps(star_ratings),
        'star_counts': json.dumps(star_counts),
        'review_dates': json.dumps(review_dates),
        'review_counts': json.dumps(review_counts),
        'most_busy_month_labels': json.dumps(most_busy_month_labels),
        'most_busy_month_values': json.dumps(most_busy_month_values),
        'sentiment_labels': json.dumps(sentiment_data['labels']),
        'sentiment_values': json.dumps(sentiment_data['values']),
        'sentiment_score': json.dumps(sentiment_data['score']),
        'monthly_avg_rating_labels':json.dumps(monthly_avg_rating_labels),
        'monthly_avg_rating_values':json.dumps(monthly_avg_rating_values),
        'top_positive_reviews_labels': json.dumps(top_positive_reviews_data['labels']),
        'top_positive_reviews_values': json.dumps(top_positive_reviews_data['values']),
        'top_negative_reviews_labels': json.dumps(top_negative_reviews_data['labels']),
        'top_negative_reviews_values': json.dumps(top_negative_reviews_data['values']),
        'recommendation': recommendation,
        'gemini_response':gemini_response,

        # Add data for other charts as needed
    }
    
    return render(request, 'review_analytics.html', context)
# https://www.amazon.in/Apple-AirPods-Generation-MagSafe-USB%E2%80%91C/product-reviews/B0CHX719JD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=1

# https://www.amazon.in/Apple-iPhone-15-128-GB/product-reviews/B0CHX1W1XY/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1

# https://www.amazon.in/Samsung-Galaxy-Smartphone-Titanium-Storage/product-reviews/B0CS5Z3T4M/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=7

# https://www.amazon.in/Sony-WH-1000XM4-Cancelling-Headphones-Bluetooth/dp/B0863TXGM3/