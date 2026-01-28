import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def run_eda():
    print("Fetching review data for EDA...")
    # Fetch essential columns to minimize data usage
    res = supabase.table('reviews').select('product_code, rating, review_content, review_date').execute()
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("No reviews found.")
        return

    print(f"\n1. Basic Stats")
    print(f"Total reviews: {len(df)}")
    print(f"Total unique books: {df['product_code'].nunique()}")
    
    # Missing Values
    print(f"\n2. Missing Values")
    print(df.isnull().sum())
    
    # Review distribution per book
    counts = df['product_code'].value_counts()
    print(f"\n3. Distribution per Book")
    print(f"Max reviews for one book: {counts.max()}")
    print(f"Min reviews for one book: {counts.min()}")
    print(f"Mean reviews: {counts.mean():.2f}")
    print(f"Median reviews: {counts.median()}")
    print(f"Books with only 1 review: {(counts == 1).sum()}")
    print(f"Books with < 5 reviews: {(counts < 5).sum()} ({(counts < 5).sum()/len(counts)*100:.1f}%)")
    
    # Time Analysis
    # Convert review_date to datetime (Format: '2025.01.20' likely)
    df['date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['year_week'] = df['date'].dt.strftime('%Y-%U')
    
    weekly_counts = df.groupby(['product_code', 'year_week']).size().reset_index(name='count')
    print(f"\n4. Weekly Inconsistency")
    print(f"Average reviews per book per week (when they occur): {weekly_counts['count'].mean():.2f}")
    
    # Continuity Check: How many weeks does a book stay active with reviews?
    continuity = weekly_counts.groupby('product_code').size()
    print(f"Mean active weeks (with reviews): {continuity.mean():.2f} weeks")
    print(f"Max active weeks: {continuity.max()} weeks")
    
    # Text Analysis
    df['text_len'] = df['review_content'].str.len().fillna(0)
    print(f"\n5. Content Quality")
    print(f"Average review length: {df['text_len'].mean():.1f} characters")
    print(f"Reviews shorter than 20 chars: {(df['text_len'] < 20).sum()} ({(df['text_len'] < 20).sum()/len(df)*100:.1f}%)")

if __name__ == "__main__":
    run_eda()
