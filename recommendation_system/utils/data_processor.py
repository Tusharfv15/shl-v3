import pandas as pd
import os
import re
from typing import List, Dict, Any

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters and lowercase
    text = text.lower()
    
    return text.strip()

def prepare_assessment_data(data_file: str) -> pd.DataFrame:
    """
    Prepare assessment data for embedding
    
    Args:
        data_file: Path to the CSV file containing assessment data
        
    Returns:
        DataFrame with processed assessment data
    """
    # Load the data
    df = pd.read_csv(data_file)
    
    # Clean up any missing values
    df = df.fillna('')
    
    # Combine relevant features into a single text field for embedding
    df['combined_text'] = (
        df['name'] + '. ' + 
        df['category'] + '. ' + 
        df['description'] + '. ' + 
        'Job levels: ' + df['job_levels'] + '. ' +
        'Test type: ' + df['test_type'] + '.'
    )
    
    # Ensure all columns have proper data types
    df['remote_testing'] = df['remote_testing'].astype(str)
    df['adaptive_irt'] = df['adaptive_irt'].astype(str)
    
    return df

def create_assessment_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create payload dictionaries for each assessment
    
    Args:
        df: DataFrame with assessment data
        
    Returns:
        List of payload dictionaries
    """
    payloads = []
    
    columns_to_include = [
        'name', 'category', 'description', 'job_levels', 
        'languages', 'assessment_length', 'remote_testing', 
        'adaptive_irt', 'test_type', 'url'
    ]
    
    for _, row in df.iterrows():
        payload = {}
        
        # Include all relevant columns in the payload
        for col in columns_to_include:
            if col in row:
                payload[col] = row[col]
        
        # Process test_type to handle multiple types
        if 'test_type' in payload and isinstance(payload['test_type'], str):
            # Split test types if they're comma separated
            test_types = [t.strip() for t in payload['test_type'].split(',')]
            payload['test_type'] = test_types[0] if len(test_types) == 1 else test_types
        
        payloads.append(payload)
    
    return payloads

def clean_test_type(test_type: str) -> List[str]:
    """
    Clean and normalize test type values
    
    Args:
        test_type: String containing test type(s)
        
    Returns:
        List of clean test types
    """
    if not test_type or not isinstance(test_type, str):
        return []
    
    # Split by commas and clean up individual types
    types = [t.strip() for t in test_type.split(',')]
    return types 