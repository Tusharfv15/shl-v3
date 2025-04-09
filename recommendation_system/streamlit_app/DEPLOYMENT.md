# Deploying to Streamlit Cloud

This guide will help you deploy the SHL Assessment Recommender to Streamlit Cloud.

## Option 1: Deploy with Qdrant Cloud (Recommended)

For the **fastest and most reliable deployment**, use Qdrant Cloud instead of local vector storage:

1. Create a Qdrant Cloud account at [https://cloud.qdrant.io/](https://cloud.qdrant.io/)

2. Create a new cluster and get your API credentials:

   - Cluster URL (e.g., `https://your-cluster-url.qdrant.io`)
   - API Key

3. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)

   - Connect to your GitHub repository
   - Set the Main app file path to:
     ```
     recommendation_system/streamlit_app/streamlit_app.py
     ```

4. Add your API keys as secrets in the app settings:

   ```
   OPENAI_API_KEY = "your-actual-openai-api-key"
   QDRANT_URL = "https://your-cluster-url.qdrant.io"
   QDRANT_API_KEY = "your-actual-qdrant-api-key"
   ```

5. Deploy!

6. After deployment, you'll need to run the embedding process once to populate your Qdrant Cloud:
   - Use the "Build Embeddings" button in the app
   - Or run locally: `python build_embeddings.py --data-file data/shl_assessments.csv`

## Option 2: Deploy the Standalone App

If you don't want to use Qdrant Cloud, use the standalone app for a simpler deployment:

1. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect to your GitHub repository
3. Set the Main app file path to:
   ```
   recommendation_system/streamlit_app/standalone_app.py
   ```
4. Add your OpenAI API key as a secret:
   ```
   OPENAI_API_KEY = "your-actual-api-key"
   ```
5. Deploy!

This standalone app contains all the code needed to run the recommender with sample data directly in Streamlit Cloud, without needing any other files or dependencies.

## Option 3: Deploy the Full App with Local Storage (Not Recommended)

If you want to deploy the full app with local vector storage:

1. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect to your GitHub repository
3. Set the Main app file path to:
   ```
   recommendation_system/streamlit_app/app.py
   ```
4. Add your OpenAI API key as a secret:
   ```
   OPENAI_API_KEY = "your-actual-openai-api-key"
   ```
5. Deploy!

Note: This approach may result in very slow deployments or timeouts as the vector database needs to be built during deployment.

## Troubleshooting

If you encounter any issues deploying:

1. Check the logs in Streamlit Cloud's app dashboard
2. Common errors:
   - **Module import errors**: Try using the standalone app instead
   - **Dependency errors**: Check if you need to update the requirements.txt file
   - **API key errors**: Make sure your API keys are correctly set in the secrets
   - **Path errors**: Try using absolute imports instead of relative imports
   - **Slow deployment**: Switch to Qdrant Cloud or use the standalone app

## Running Locally

To run the app locally:

```bash
cd recommendation_system
streamlit run streamlit_app/standalone_app.py
```

For the full app:

```bash
cd recommendation_system
python run_app.py
```
