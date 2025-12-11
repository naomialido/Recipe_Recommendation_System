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
from werkzeug.utils import secure_filename
import base64
import io

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
    print("=" * 60)
    print("Loading Recipe Database...")
    print("=" * 60)
    
    try:
        df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
        print(f"✅ Dataset loaded: {len(df)} recipes found")
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return False

    if len(df) > MAX_RECIPES:
        print(f"📊 Limiting to {MAX_RECIPES} recipes for optimization")
        df = df.head(MAX_RECIPES)

    recipes = df.to_dict(orient="records")

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(PROCESSED_RECIPES_FILE):
        try:
            print("📂 Loading FAISS index...")
            index = faiss.read_index(FAISS_INDEX_FILE)
            
            print("📂 Loading processed recipes...")
            with open(PROCESSED_RECIPES_FILE, 'rb') as f:
                recipes = pickle.load(f)
            
            print(f"✅ Successfully loaded {len(recipes)} recipes with embeddings!")
            print("=" * 60)
            gc.collect()
            return True
            
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            print("💡 Please run generate_embeddings.py first!")
            return False
    else:
        print("❌ ERROR: FAISS index not found.")
        print("💡 Please run: python generate_embeddings.py")
        print("=" * 60)
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "recipes_loaded": len(recipes),
        "max_recipes": MAX_RECIPES,
        "faiss_index_loaded": index is not None
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
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_input_text]
        )
        user_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Search FAISS index
        distances, indices = index.search(user_embedding, 10)

        # Build recommendations
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
                'matchScore': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
            })

        gc.collect()
        return jsonify({
            'recommendations': recommendations,
            'totalFound': len(recommendations)
        })

    except Exception as e:
        print(f"Error in recommend_recipes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend-ai', methods=['POST'])
def recommend_ai_recipes():
    """AI-enhanced recommendations with analysis"""
    try:
        data = request.get_json()
        user_ingredients = data.get('ingredients', [])

        if not user_ingredients:
            return jsonify({"error": "No ingredients provided"}), 400

        if index is None or len(recipes) == 0:
            return jsonify({"error": "Recipe database not loaded"}), 500

        print(f"🤖 AI recommendation for: {', '.join(user_ingredients)}")

        # Get vector search results first
        user_input_text = "Ingredients: " + ", ".join(user_ingredients)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_input_text]
        )
        user_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Search for top 5 matches
        distances, indices = index.search(user_embedding, 5)

        # Build recommendations
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
                'ingredients': str(recipe.get('Ingredients_Raw', ''))[:500],  # Truncate for token limit
                'instructions': str(recipe.get('Instructions', ''))[:300],
                'servings': str(recipe.get('Servings', 'N/A')),
                'rating': str(recipe.get('Rating Value', 'N/A')),
                'url': str(recipe.get('URL', '')),
                'matchedIngredients': matched,
                'matchScore': float(1.0 / (1.0 + distances[0][i]))
            })

        # Create concise summary for AI
        recipes_summary = "\n".join([
            f"• {r['name']} ({r['cuisine']}) - Prep: {r['prepTime']}, Cook: {r['cookTime']}"
            for r in recommendations
        ])

        # Build AI prompt
        prompt = f"""You have these ingredients: {', '.join(user_ingredients)}

Here are your top recipe matches:
{recipes_summary}

Provide a brief, friendly analysis (2-3 sentences max):
1. Which recipe looks easiest/quickest
2. Any common ingredients they might need to add
3. A quick recommendation

Keep it super concise and helpful!"""

        # Get AI analysis
        try:
            print("🤖 Calling OpenAI for analysis...")
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful recipe assistant. Keep responses very brief (2-3 sentences), friendly, and practical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
            ai_analysis = ai_response.choices[0].message.content
            print(f"✅ AI analysis generated: {len(ai_analysis)} chars")
            
        except Exception as e:
            print(f"❌ AI error: {e}")
            ai_analysis = "AI analysis temporarily unavailable, but here are your recipe matches!"

        gc.collect()
        
        return jsonify({
            'recommendations': recommendations,
            'aiAnalysis': ai_analysis,
            'totalFound': len(recommendations)
        })

    except Exception as e:
        print(f"❌ Error in recommend_ai_recipes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze an image to detect ingredients using OpenAI Vision"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Read image file and convert to base64
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Call OpenAI Vision API
        print(f"🖼️ Analyzing image: {image_file.filename}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and list all visible food ingredients or ingredients you can identify. Return ONLY a simple comma-separated list of ingredient names in lowercase, nothing else. Example: 'chicken, tomatoes, garlic, onion'. If no food ingredients are visible, return 'no ingredients found'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        # Parse the response
        analysis = response.choices[0].message.content.strip().lower()
        
        if 'no ingredients' in analysis:
            print("⚠️ No ingredients detected in image")
            return jsonify({
                "ingredients": [],
                "message": "No food ingredients detected in the image. Try uploading a clearer food image."
            })

        # Parse comma-separated ingredients
        ingredients_list = [ing.strip() for ing in analysis.split(',') if ing.strip()]
        ingredients_list = [ing for ing in ingredients_list if len(ing) > 1]  # Filter out single characters

        print(f"✅ Detected ingredients: {ingredients_list}")
        
        gc.collect()
        
        return jsonify({
            "ingredients": ingredients_list,
            "message": f"Successfully detected {len(ingredients_list)} ingredient(s)"
        })

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🍳 RECIPE RECOMMENDATION API")
    print("=" * 60)
    
    success = load_embeddings_only()
    
    if not success:
        print("\n❌ STARTUP FAILED!")
        print("Please ensure:")
        print("1. dataset.csv exists in data/ folder")
        print("2. Run: python generate_embeddings.py")
        print("3. OpenAI API key is set in .env")
        print("=" * 60 + "\n")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✅ SERVER READY!")
    print(f"📊 Loaded: {len(recipes)} recipes")
    print(f"🔍 FAISS index: Active")
    print(f"🌐 Running on: http://0.0.0.0:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)