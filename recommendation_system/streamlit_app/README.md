# Deploying to Streamlit Cloud

This folder contains everything needed to deploy the SHL Assessment Recommender to Streamlit Cloud.

## Deployment Steps

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud) if you don't have one

2. Create a new app and connect to your GitHub repository

3. Configure the app with the following settings:

   - **Main file path**: `recommendation_system/streamlit_app/app.py`
   - **Python version**: 3.9 (or latest stable)

4. Add your OpenAI API key as a secret:

   - In the app settings, find "Secrets"
   - Add the following secret:
     ```
     OPENAI_API_KEY = "your_actual_openai_api_key"
     ```

5. Deploy!

## Troubleshooting

If you encounter any dependency issues, you can try:

1. Updating the requirements.txt file with more relaxed version constraints
2. Check Streamlit Cloud logs for specific error messages
3. Consider adding any missing system-level dependencies to packages.txt

## Local Development

To run this app locally:

```bash
cd recommendation_system
python run_app.py
```

Or directly:

```bash
cd recommendation_system
streamlit run streamlit_app/app.py
```
 