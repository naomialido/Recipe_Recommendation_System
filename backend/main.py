from dotenv import load_dotenv
# import os
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#load env and openai client
load_dotenv()
client = OpenAI()

#load dataset
df = pd.read_csv("./data/dataset.csv", sep=";", encoding="latin1")
recipes = df.to_dict(orient="records")

#recipe embeddings
print("Generating recipe embeddings...")
recipe_texts = [
    f"{r['Name']}. Ingredients: {r['Ingredients_Raw']}. Instructions: {r['Instructions']}"
    for r in recipes
]

#batch embeddings to avoid huge requests
recipe_embeddings = []
batch_size = 50
for i in range(0, len(recipe_texts), batch_size):
    batch = recipe_texts[i:i+batch_size]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )
    recipe_embeddings.extend([item.embedding for item in response.data])

recipe_embeddings = np.array(recipe_embeddings)
print("Embeddings ready.\n")

#user input
print("Enter your ingredients one at a time.")
print("Type 'q' and press ENTER when done.\n")

user_ingredients = []

while True:
    item = input("Ingredient: ").strip().lower()
    if item == "q":
        break  # Exit loop when user is finished
    if item:
        user_ingredients.append(item)

if not user_ingredients:
    print("\nNo ingredients provided. Exiting.")
    exit()

#cosine similarity
user_input_text = "Ingredients: " + ", ".join(user_ingredients)

user_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_input_text
).data[0].embedding

similarities = cosine_similarity([user_embedding], recipe_embeddings)[0]
top_indices = similarities.argsort()[-5:][::-1]  # top 5 matches

#prompt for openai gpt
top_recipes_text = ""
for idx in top_indices:
    r = recipes[idx]
    top_recipes_text += (
        f"Name: {r['Name']}\n"
        f"Ingredients: {r['Ingredients_Raw']}\n"
        f"Instructions: {r['Instructions']}\n"
        f"Cuisine: {r['Cuisine']}\n"
        f"Category: {r['Category']}\n"
        "---------------------------------------\n"
    )

prompt = f"""
You are a recipe recommendation engine.

RULES:
- ONLY recommend recipes from the list provided below.
- NEVER invent new recipes.
- Rank recipes by how many of the user's ingredients they contain.
- Explain why each recipe is a match.

USER INGREDIENTS:
{', '.join(user_ingredients)}

CANDIDATE RECIPES:
{top_recipes_text}
"""

#gpt recommendation
response = client.responses.create(
    model="gpt-4o",
    input=prompt
)

#display results
print("\n================== RECOMMENDATIONS ==================\n")
print(response.output_text)
print("\n=====================================================\n")
