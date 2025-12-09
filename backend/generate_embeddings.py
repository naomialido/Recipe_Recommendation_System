import os, pickle, gc
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "recipe_embeddings.npy")
PROCESSED_RECIPES_FILE = os.path.join(DATA_DIR, "processed_recipes.pkl")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "recipe_index.faiss")

MAX_RECIPES = 10000

def embed_batch(batch):
    response = client.embeddings.create(model="text-embedding-3-small", input=batch)
    return [item.embedding for item in response.data]

def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
    if len(df) > MAX_RECIPES:
        df = df.head(MAX_RECIPES)
    recipes = df.to_dict(orient="records")

    recipe_texts = []
    for r in recipes:
        text = f"{r.get('Name','')}. Cuisine: {r.get('Cuisine','')}. Category: {r.get('Category','')}. Ingredients: {r.get('Ingredients_Raw','')}. Instructions: {r.get('Instructions','')}"
        recipe_texts.append(text)

    embeddings_list = []
    batch_size = 10
    for i in range(0, len(recipe_texts), batch_size):
        batch = recipe_texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size+1}/{len(recipe_texts)//batch_size+1}...")
        embeddings_list.extend(embed_batch(batch))

    recipe_embeddings = np.array(embeddings_list, dtype=np.float32)

    #Save files
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(EMBEDDINGS_FILE, recipe_embeddings)
    with open(PROCESSED_RECIPES_FILE, "wb") as f:
        pickle.dump(recipes, f, protocol=4)

    # uild FAISS index
    d = recipe_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(recipe_embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)

    print("✅ Embeddings and FAISS index generated successfully!")

if __name__ == "__main__":
    main()
