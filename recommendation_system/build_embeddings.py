import os
import argparse
from utils.data_processor import prepare_assessment_data, create_assessment_payloads
from utils.vectorize import batch_get_embeddings
from utils.vector_store import QdrantVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def build_embeddings(data_file, collection_name):
    """
    Build and store assessment embeddings
    
    Args:
        data_file: Path to the CSV file containing assessment data
        collection_name: Name of the vector collection to create
    """
    # Prepare assessment data
    print(f"Loading and processing assessment data from {data_file}...")
    df = prepare_assessment_data(data_file)
    print(f"Loaded {len(df)} assessments")
    
    # Create embeddings for the assessments
    print("Generating embeddings (this may take a while)...")
    embeddings = batch_get_embeddings(df['combined_text'].tolist())
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create assessment payloads
    print("Creating assessment payloads...")
    payloads = create_assessment_payloads(df)
    
    # Initialize vector store
    print(f"Initializing vector store '{collection_name}'...")
    vector_store = QdrantVectorStore(collection_name=collection_name)
    
    # Add vectors to the store
    print("Adding vectors to the store...")
    vector_store.add_vectors(embeddings, payloads)
    
    print("Embedding build process completed successfully!")
    return vector_store

def main():
    parser = argparse.ArgumentParser(description="Build SHL assessment embeddings")
    parser.add_argument("--data-file", type=str, default="data/shl_assessments_first_row.csv",
                      help="Path to the CSV file containing assessment data")
    parser.add_argument("--collection-name", type=str, default="shl_assessments",
                      help="Name of the vector collection to create")
    
    args = parser.parse_args()
    
    build_embeddings(args.data_file, args.collection_name)

if __name__ == "__main__":
    main() 