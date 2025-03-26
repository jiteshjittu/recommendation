from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import os

app = Flask(__name__)

# Global variable to store the model
item_similarity_df = None

# Load the model
def load_model():
    global item_similarity_df
    try:
        # Check if model file exists
        if not os.path.exists('model.pkl'):
            print("Model file not found. Creating new model...")
            # If model doesn't exist, create it with reduced size
            df = pd.read_csv('Amazon Beauty Recommendation system.csv')
            
            # Reduce to top 300 most popular products and active users
            # This will create a 300x300 similarity matrix which should be under 25MB
            popular_products = df['ProductId'].value_counts().nlargest(300).index
            active_users = df['UserId'].value_counts().nlargest(300).index
            
            # Filter the dataframe
            df_filtered = df[df['ProductId'].isin(popular_products) & df['UserId'].isin(active_users)]
            
            # Create pivot table
            pivot_table = df_filtered.pivot_table(
                index='UserId', 
                columns='ProductId', 
                values='Rating', 
                aggfunc='mean', 
                fill_value=0
            )
            
            # Calculate similarity matrix
            item_similarity = cosine_similarity(pivot_table.T)
            item_similarity_df = pd.DataFrame(
                item_similarity, 
                index=pivot_table.columns, 
                columns=pivot_table.columns
            )
            
            # Save the reduced model
            joblib.dump(item_similarity_df, 'model.pkl')
            print("New model created and saved successfully.")
        else:
            print("Loading existing model...")
            item_similarity_df = joblib.load('model.pkl')
            print("Model loaded successfully.")
            
        return item_similarity_df
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_similar_items(product_id, top_n=5):
    if item_similarity_df is None:
        return None
    if product_id not in item_similarity_df.index:
        return None
    similar_items = item_similarity_df[product_id].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_items

@app.route('/')
def home():
    try:
        global item_similarity_df
        if item_similarity_df is None:
            item_similarity_df = load_model()
        
        if item_similarity_df is None:
            return render_template('home.html', 
                                 available_products=[], 
                                 error="Error loading model. Please try again later.")
        
        # Get list of available product IDs for the dropdown
        available_products = sorted(item_similarity_df.index.tolist())
        return render_template('home.html', available_products=available_products)
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        return render_template('home.html', 
                             available_products=[], 
                             error="An error occurred. Please try again later.")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        product_id = request.form['product_id']
        similar_items = get_similar_items(product_id)
        
        if similar_items is None:
            return render_template('result.html', 
                                 error="Product ID not found. Please try another product ID.",
                                 product_id=product_id)
        
        recommendations = []
        for idx, similarity in similar_items.items():
            recommendations.append({
                'product_id': idx,
                'rating': round(similarity, 2)
            })
        
        return render_template('result.html', 
                             recommendations=recommendations,
                             product_id=product_id)
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return render_template('result.html', 
                             error="An error occurred. Please try again later.",
                             product_id=product_id)

if __name__ == '__main__':
    # Load model before starting the server
    item_similarity_df = load_model()
    app.run(debug=True) 