from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import pickle
import json
import gc
import faiss   

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  #Enable CORS for frontend requests

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#global variables
recipes = []
index = None   # FAISS index replaces recipe_embeddings
EMBEDDINGS_FILE = '../data/recipe_embeddings.npy'
PROCESSED_RECIPES_FILE = '../data/processed_recipes.pkl'
FAISS_INDEX_FILE = '../data/recipe_index.faiss'

MAX_RECIPES = 10000

def build_faiss_index(embeddings):
    """Build and save FAISS index"""
    d = embeddings.shape[1]  # embedding dimension
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    print(f"FAISS index saved with {faiss_index.ntotal} vectors.")
    return faiss_index

def load_or_generate_embeddings():
    """Load pre-computed embeddings or generate them if they don't exist"""
    global recipes, index
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

    print("Looking for dataset at:", DATA_PATH)
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
        print(f"Dataset loaded: {len(df)} recipes found")
    except FileNotFoundError:
        print("ERROR: dataset.csv not found at", DATA_PATH)
        return False
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False
    
    # Limit dataset size for memory
    if len(df) > MAX_RECIPES:
        print(f"Limiting to {MAX_RECIPES} recipes for memory optimization...")
        df = df.head(MAX_RECIPES)
    
    recipes = df.to_dict(orient="records")
    
    # Check if FAISS index already exists
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(PROCESSED_RECIPES_FILE):
        print("Loading FAISS index and recipes...")
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(PROCESSED_RECIPES_FILE, 'rb') as f:
                recipes = pickle.load(f)
            print(f"Loaded {len(recipes)} recipes with FAISS index!")
            gc.collect()
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Will generate new embeddings...")
    
    # Generate embeddings
    print(f"Generating recipe embeddings for {len(recipes)} recipes...")
    recipe_texts = []
    for i, r in enumerate(recipes):
        text = f"{r.get('Name', '')}. "
        text += f"Cuisine: {r.get('Cuisine', '')}. "
        text += f"Category: {r.get('Category', '')}. "
        text += f"Ingredients: {r.get('Ingredients_Raw', '')}. "
        text += f"Instructions: {r.get('Instructions', '')}"
        recipe_texts.append(text)
    
    embeddings_list = []
    batch_size = 50
    total_batches = (len(recipe_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(recipe_texts), batch_size):
        batch = recipe_texts[i:i+batch_size]
        batch_num = i//batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings_list.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            embeddings_list.extend([[0] * 1536 for _ in batch])
    
    recipe_embeddings = np.array(embeddings_list, dtype=np.float32)
    
    # Save embeddings + recipes
    print("Saving embeddings and recipes...")
    os.makedirs('../data', exist_ok=True)
    np.save(EMBEDDINGS_FILE, recipe_embeddings)
    with open(PROCESSED_RECIPES_FILE, 'wb') as f:
        pickle.dump(recipes, f, protocol=4)
    
    # Build FAISS index
    index = build_faiss_index(recipe_embeddings)
    gc.collect()
    return True

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "recipes_loaded": len(recipes),
        "max_recipes": MAX_RECIPES
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_recipes():
    try:
        data = request.get_json()
        user_ingredients = data.get('ingredients', [])
        
        if not user_ingredients:
            return jsonify({"error": "No ingredients provided"}), 400
        
        if index is None or len(recipes) == 0:
            return jsonify({"error": "Recipe database not loaded"}), 500
        
        # Create embedding for user input
        user_input_text = "Ingredients: " + ", ".join(user_ingredients)
        user_embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_input_text
        )
        user_embedding = np.array([user_embedding_response.data[0].embedding], dtype=np.float32)
        
        # Query FAISS index
        distances, indices = index.search(user_embedding, 10)
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            recipe = recipes[idx]
            recipe_ingredients = str(recipe.get('Ingredients_Raw', '')).lower()
            matched = [ing for ing in user_ingredients if ing.lower() in recipe_ingredients]
            
            recommendations.append({
                'name': str(recipe.get('Name', 'Unknown')),
                'cuisine': str(recipe.get('Cuisine', 'Unknown')),
                'category': str(recipe.get('Category', 'Unknown')),
                'prepTime': str(recipe.get('Preparation Time', 'N/A')),
                'cookTime': str(recipe.get('Cooking Time', 'N/A')),
                'ingredients': str(recipe.get('Ingredients_Raw', '')),
                'instructions': str(recipe.get('Instructions', '')),
                'servings': str(recipe.get('Servings', 'N/A')),
                'rating': str(recipe.get('Rating Value', 'N/A')),
                'url': str(recipe.get('URL', '')),
                'matchedIngredients': matched,
                'matchScore': float(distances[0][i])
            })
        
        gc.collect()
        return jsonify({'recommendations': recommendations, 'totalFound': len(recommendations)})
        
    except Exception as e:
        print(f"Error in recommend_recipes: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Flask server with FAISS...")
    print(f"Max recipes configured: {MAX_RECIPES}")
    print("=" * 60)
    
    success = load_or_generate_embeddings()
    if not success:
        print("ERROR: Failed to load recipe data!")
        exit(1)
    
    print("Server ready!")
    print(f"Loaded {len(recipes)} recipes into FAISS index")
    app.run(debug=False, host='0.0.0.0', port=5000)
