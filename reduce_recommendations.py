import pandas as pd
import numpy as np

def analyze_ratings(input_file):
    # Read the first chunk to understand rating distribution
    chunk = pd.read_csv(input_file, nrows=10000)
    print("\nRating distribution:")
    print(chunk['Rating'].value_counts().sort_index())
    print("\nRating range:")
    print(f"Min rating: {chunk['Rating'].min()}")
    print(f"Max rating: {chunk['Rating'].max()}")
    return chunk['Rating'].max()

def reduce_recommendations(input_file, output_file):
    # First analyze the ratings
    max_rating = analyze_ratings(input_file)
    
    # Adjust the rating threshold to be 80% of max rating
    rating_threshold = max_rating * 0.8
    print(f"\nUsing rating threshold of {rating_threshold}")
    
    # Read the CSV file in chunks
    chunk_size = 100000
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    # Initialize empty list to store filtered chunks
    filtered_chunks = []
    
    # Process each chunk
    for chunk in chunks:
        # Filter for high ratings (top 20%)
        filtered_chunk = chunk[chunk['Rating'] > rating_threshold].copy()
        
        if not filtered_chunk.empty:
            # Group by product and ensure exactly 5 recommendations per product
            filtered_chunk = filtered_chunk.groupby('ProductId').head(5)
            filtered_chunks.append(filtered_chunk)
    
    if not filtered_chunks:
        print("No products found matching the criteria!")
        return
        
    # Combine all filtered chunks
    reduced_df = pd.concat(filtered_chunks, ignore_index=True)
    
    # Save to new CSV file
    reduced_df.to_csv(output_file, index=False)
    print(f"\nOriginal file processed and reduced to {len(reduced_df)} rows")
    print(f"Number of unique products: {reduced_df['ProductId'].nunique()}")

if __name__ == "__main__":
    input_file = "Amazon Beauty Recommendation system.csv"
    output_file = "reduced_recommendations.csv"
    reduce_recommendations(input_file, output_file) 