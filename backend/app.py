from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import pickle
import gc
import faiss

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

recipes = []
index = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
PROCESSED_RECIPES_FILE = os.path.join(DATA_DIR, "processed_recipes.pkl")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "recipe_index.faiss")

MAX_RECIPES = 10000

def load_embeddings_only():
    global recipes, index
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
    except Exception as e:
        print("ERROR loading dataset:", e)
        return False

    if len(df) > MAX_RECIPES:
        df = df.head(MAX_RECIPES)

    recipes = df.to_dict(orient="records")

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(PROCESSED_RECIPES_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(PROCESSED_RECIPES_FILE, 'rb') as f:
                recipes = pickle.load(f)
            return True
        except Exception as e:
            print("Error loading FAISS index:", e)
            return False
    else:
        print("ERROR: FAISS index not found.")
        return False

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

        user_input_text = "Ingredients: " + ", ".join(user_ingredients)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_input_text]
        )
        user_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        distances, indices = index.search(user_embedding, 10)

        recommendations = []
        for i, idx in enumerate(indices[0]):
            recipe = recipes[idx]
            matched = [ing for ing in user_ingredients if ing.lower() in str(recipe.get('Ingredients_Raw', '')).lower()]
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
        print("Error in recommend_recipes:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend-ai', methods=['POST'])
def recommend_ai_recipes():
    try:
        data = request.get_json()
        # Accept either prompt or ingredients
        user_prompt = data.get('prompt') or "Ingredients: " + ", ".join(data.get('ingredients', []))

        if not user_prompt.strip():
            return jsonify({"error": "No prompt provided"}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful recipe assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )

        ai_text = response.choices[0].message.content
        return jsonify({"aiAnalysis": ai_text})

    except Exception as e:
        print("Error in recommend_ai_recipes:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    success = load_embeddings_only()
    if not success:
        exit(1)
    app.run(debug=False, host='0.0.0.0', port=5000)
