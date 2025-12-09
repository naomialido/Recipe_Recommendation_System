from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables
recipes = []
recipe_embeddings = None
EMBEDDINGS_FILE = 'data/recipe_embeddings.pkl'
PROCESSED_RECIPES_FILE = 'data/processed_recipes.pkl'

def load_or_generate_embeddings():
    """Load pre-computed embeddings or generate them if they don't exist"""
    global recipes, recipe_embeddings
    
    print("Loading dataset...")
    df = pd.read_csv("./data/dataset.csv", sep=";", encoding="latin1")
    recipes = df.to_dict(orient="records")
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(PROCESSED_RECIPES_FILE):
        print("Loading pre-computed embeddings...")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            recipe_embeddings = pickle.load(f)
        with open(PROCESSED_RECIPES_FILE, 'rb') as f:
            recipes = pickle.load(f)
        print(f"Loaded {len(recipes)} recipes with embeddings!")
        return
    
    #Generate embeddings
    print("Generating recipe embeddings... This will take a while but only happens once!")
    recipe_texts = []
    
    for r in recipes:
        # Create text representation of recipe
        text = f"{r['Name']}. "
        text += f"Cuisine: {r['Cuisine']}. "
        text += f"Category: {r['Category']}. "
        text += f"Ingredients: {r['Ingredients_Raw']}. "
        text += f"Instructions: {r['Instructions']}"
        recipe_texts.append(text)
    
    #Generate embeddings in batches
    embeddings_list = []
    batch_size = 100
    total_batches = (len(recipe_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(recipe_texts), batch_size):
        batch = recipe_texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batches}...")
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings_list.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed batch
            embeddings_list.extend([[0] * 1536 for _ in batch])
    
    recipe_embeddings = np.array(embeddings_list)
    
    #Save embeddings for future use
    print("Saving embeddings to disk...")
    os.makedirs('data', exist_ok=True)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(recipe_embeddings, f)
    with open(PROCESSED_RECIPES_FILE, 'wb') as f:
        pickle.dump(recipes, f)
    
    print("Embeddings generated and saved successfully!")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "recipes_loaded": len(recipes)})

@app.route('/api/recommend', methods=['POST'])
def recommend_recipes():
    """Get recipe recommendations based on user ingredients"""
    try:
        data = request.get_json()
        user_ingredients = data.get('ingredients', [])
        
        if not user_ingredients:
            return jsonify({"error": "No ingredients provided"}), 400
        
        #Create embedding for user input
        user_input_text = "Ingredients: " + ", ".join(user_ingredients)
        
        user_embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_input_text
        )
        user_embedding = np.array([user_embedding_response.data[0].embedding])
        
        #Calculate cosine similarities
        similarities = cosine_similarity(user_embedding, recipe_embeddings)[0]
        
        #Get top 10 matches
        top_indices = similarities.argsort()[-10:][::-1]
        
        #Prepare recommendations
        recommendations = []
        for idx in top_indices:
            recipe = recipes[idx]
            
            #Calculate ingredient match
            recipe_ingredients = recipe.get('Ingredients_Raw', '').lower()
            matched = [ing for ing in user_ingredients if ing.lower() in recipe_ingredients]
            
            recommendations.append({
                'name': recipe.get('Name', 'Unknown'),
                'cuisine': recipe.get('Cuisine', 'Unknown'),
                'category': recipe.get('Category', 'Unknown'),
                'prepTime': recipe.get('Preparation Time', 'N/A'),
                'cookTime': recipe.get('Cooking Time', 'N/A'),
                'ingredients': recipe.get('Ingredients_Raw', ''),
                'instructions': recipe.get('Instructions', ''),
                'servings': recipe.get('Servings', 'N/A'),
                'rating': recipe.get('Rating Value', 'N/A'),
                'url': recipe.get('URL', ''),
                'matchedIngredients': matched,
                'matchScore': float(similarities[idx])
            })
        
        return jsonify({
            'recommendations': recommendations,
            'totalFound': len(recommendations)
        })
        
    except Exception as e:
        print(f"Error in recommend_recipes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend-ai', methods=['POST'])
def recommend_with_ai():
    """Get AI-enhanced recipe recommendations"""
    try:
        data = request.get_json()
        user_ingredients = data.get('ingredients', [])
        
        if not user_ingredients:
            return jsonify({"error": "No ingredients provided"}), 400
        
        #First get vector search results
        user_input_text = "Ingredients: " + ", ".join(user_ingredients)
        user_embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_input_text
        )
        user_embedding = np.array([user_embedding_response.data[0].embedding])
        similarities = cosine_similarity(user_embedding, recipe_embeddings)[0]
        top_indices = similarities.argsort()[-5:][::-1]
        
        #Get top recipes
        top_recipes = []
        for idx in top_indices:
            recipe = recipes[idx]
            top_recipes.append({
                'name': recipe.get('Name'),
                'cuisine': recipe.get('Cuisine'),
                'category': recipe.get('Category'),
                'ingredients': recipe.get('Ingredients_Raw'),
                'instructions': recipe.get('Instructions'),
                'prepTime': recipe.get('Preparation Time'),
                'cookTime': recipe.get('Cooking Time')
            })
        
        #Use GPT to analyze and present recommendations
        prompt = f"""
You are a recipe recommendation assistant. Based on the user's ingredients and these matching recipes from our database, provide helpful recommendations.

USER INGREDIENTS: {', '.join(user_ingredients)}

TOP MATCHING RECIPES:
{json.dumps(top_recipes, indent=2)}

Provide a brief analysis explaining:
1. Why each recipe matches the user's ingredients
2. Which recipe would be easiest to make
3. Any additional ingredients they might need

Keep your response concise and friendly.
"""
        
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_analysis = chat_response.choices[0].message.content
        
        return jsonify({
            'recommendations': top_recipes,
            'aiAnalysis': ai_analysis
        })
        
    except Exception as e:
        print(f"Error in recommend_with_ai: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recipe/<int:recipe_id>', methods=['GET'])
def get_recipe_details(recipe_id):
    """Get detailed information about a specific recipe"""
    try:
        if recipe_id < 0 or recipe_id >= len(recipes):
            return jsonify({"error": "Recipe not found"}), 404
        
        recipe = recipes[recipe_id]
        return jsonify(recipe)
        
    except Exception as e:
        print(f"Error in get_recipe_details: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    load_or_generate_embeddings()
    print("Server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)