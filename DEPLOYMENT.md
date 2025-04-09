# SHL Assessment Recommender API Deployment Guide

This guide explains how to deploy the SHL Assessment Recommender API to Render (a free cloud hosting service).

## Deploying to Render

1. **Create a Render account**

   Go to [render.com](https://render.com/) and sign up for a free account.

2. **Create a new Web Service**

   - Click on "New" in the top-right corner
   - Select "Web Service"
   - Connect your GitHub repository (or use Render's built-in Git deployment)
   - Select the repository containing this project

3. **Configure deployment settings**

   - **Name**: `shl-recommender-api` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn run:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

4. **Add environment variables**

   Add the following environment variables in the Render dashboard:

   - `OPENAI_API_KEY`: Your OpenAI API key
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `QDRANT_URL`: Your Qdrant Cloud URL

5. **Deploy**

   Click on "Create Web Service" and wait for the deployment to complete.

After deployment, your API will be available at a URL like:
`https://shl-recommender-api.onrender.com`

## Accessing the API

Once deployed, your evaluator can access the API using:

```
GET https://shl-recommender-api.onrender.com/recommend?query=Looking%20for%20Java%20developers%20with%20business%20collaboration%20skills&top_k=5&enhanced=true
```

You can also use the POST method with a JSON body as described in the API_USAGE.md file.

## API Documentation

The API documentation will be available at:
`https://shl-recommender-api.onrender.com/docs`

## Environment Variables

Ensure these environment variables are set for the deployed service:

- `OPENAI_API_KEY`: Required for enhanced query processing
- `QDRANT_API_KEY`: Required for vector database access
- `QDRANT_URL`: Required for vector database access

## Troubleshooting

1. **API returns 500 error**:

   - Check the Render logs for error details
   - Verify your environment variables are correct
   - Ensure Qdrant collection exists and is properly set up

2. **Cold start issues**:

   - The free tier on Render spins down after inactivity
   - The first request after inactivity may take up to 30 seconds to respond

3. **Memory limitations**:
   - Free tier has memory limitations
   - Consider upgrading to paid tier for production workloads
