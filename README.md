# Recipe_Recommendation_System
A web application where users can get recipe recommendations based on ingredients they input.
## Features
1. **Health Check** – Verify that the API and recipe database are loaded.  
2. **Recipe Recommendations** – Get recipe matches based on user ingredients.  
3. **AI-Enhanced Recommendations** – Receive concise analysis and suggestions powered by OpenAI.  
4. **Spell Correction** – Automatically corrects misspelled ingredients using fuzzy matching.
5. **Ingredient Analyzer from Image** - Detects ingredients from image file uploaded
##  Requirements (refer to requirements.txt)
- Python 3.9+  
- [pip](https://pip.pypa.io/en/stable/)  
- Dependencies listed in `requirements.txt` (Flask, Flask-CORS, pandas, numpy, faiss, openai, python-dotenv, etc.)  
- OpenAI API key  

## Setup Instructions
1. Install dependencies with requirements.txt in backend folder
2. Set up environment variables in a `.env` file in the backend folder:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Generate embeddings for the recipe database by running:
   ```
   python generate_embeddings.py
   ```
4. Start the Flask backend server:
   ```
    python app.py
    ```
5. Open the `index.html` file in the frontend folder in your web browser to access the application. It will automaticall connect to the backend API at 'http://127.0.0.1:5000/api' when running locally.

## Notes
- Ensure you have a valid OpenAI API key for AI-enhanced features.
- The recipe database should be pre-loaded with recipes and their embeddings for optimal performance.
- The frontend is a simple HTML/JS interface that interacts with the Flask backend API.
