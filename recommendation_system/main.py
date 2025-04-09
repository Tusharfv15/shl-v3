import os
import argparse
import json
import requests
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
from utils.data_processor import prepare_assessment_data, create_assessment_payloads
from utils.vectorize import batch_get_embeddings, get_embedding
from utils.vector_store import QdrantVectorStore
from models.recommender import BaseRecommender

# Load environment variables
load_dotenv()

# Define SHLRecommender class for use in Streamlit app
class SHLRecommender:
    def __init__(self, collection_name="shl_assessments"):
        """Initialize the SHL Recommender with a vector store"""
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        self.base_recommender = BaseRecommender(vector_store=self.vector_store)
    
    def recommend(self, query: str, top_k: int = 10, enhanced: bool = False, 
                  filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Recommend assessments based on a text query
        
        Args:
            query: The search query or job description
            top_k: Number of recommendations to return
            enhanced: Whether to use enhanced query processing with GPT
            filters: Dict of filters to apply (remote_testing, adaptive_irt, test_type)
            
        Returns:
            List of assessment recommendations
        """
        # Extract filter parameters
        remote_testing = filters.get("remote_testing") if filters else None
        adaptive_irt = filters.get("adaptive_irt") if filters else None
        test_types = filters.get("test_type") if filters else None
        
        # Get recommendations
        if enhanced:
            recommendations = self.base_recommender.enhanced_recommendations(
                query,
                remote_testing=remote_testing,
                adaptive_irt=adaptive_irt,
                test_types=test_types,
                limit=top_k
            )
        else:
            recommendations = self.base_recommender.process_query(
                query,
                remote_testing=remote_testing,
                adaptive_irt=adaptive_irt,
                test_types=test_types,
                limit=top_k
            )
        
        return recommendations
    
    def recommend_from_url(self, url: str, top_k: int = 10, enhanced: bool = False,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Recommend assessments based on a job description URL
        
        Args:
            url: URL of the job description
            top_k: Number of recommendations to return
            enhanced: Whether to use enhanced query processing with GPT
            filters: Dict of filters to apply (remote_testing, adaptive_irt, test_type)
            
        Returns:
            List of assessment recommendations
        """
        # Fetch content from URL
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            text_content = response.text
            
            # Extract text from HTML if needed
            # This is a simple approach - could use BeautifulSoup for better extraction
            from html import unescape
            import re
            
            # Basic HTML tag removal
            text_content = re.sub(r'<[^>]+>', ' ', text_content)
            # Decode HTML entities
            text_content = unescape(text_content)
            # Remove extra whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Use the extracted text for recommendations
            return self.recommend(text_content, top_k, enhanced, filters)
            
        except Exception as e:
            raise Exception(f"Error fetching job description from URL: {e}")

# Original functions
def build_embeddings(data_file, collection_name):
    """Build and store assessment embeddings"""
    # Prepare assessment data
    df = prepare_assessment_data(data_file)
    print(f"Loaded {len(df)} assessments")
    
    # Create embeddings for the assessments
    print("Generating embeddings...")
    embeddings = batch_get_embeddings(df['combined_text'].tolist())
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create assessment payloads
    payloads = create_assessment_payloads(df)
    
    # Initialize vector store
    print(f"Initializing vector store '{collection_name}'...")
    vector_store = QdrantVectorStore(collection_name=collection_name)
    
    # Add vectors to the store
    print("Adding vectors to the store...")
    vector_store.add_vectors(embeddings, payloads)
    
    print("Embedding build process completed successfully!")
    return vector_store

def recommend(vector_store, query, job_url=None, remote_testing=None, 
             adaptive_irt=None, test_types=None, limit=10, use_enhanced=False):
    """Generate recommendations for a query"""
    recommender = BaseRecommender(vector_store=vector_store)
    
    if use_enhanced:
        return recommender.enhanced_recommendations(
            query,
            job_description_url=job_url,
            remote_testing=remote_testing,
            adaptive_irt=adaptive_irt,
            test_types=test_types,
            limit=limit
        )
    else:
        return recommender.process_query(
            query,
            remote_testing=remote_testing,
            adaptive_irt=adaptive_irt,
            test_types=test_types,
            limit=limit
        )

def format_recommendations(recommendations):
    """Format recommendations for display"""
    formatted = []
    for i, rec in enumerate(recommendations):
        formatted.append({
            "rank": i + 1,
            "name": rec['name'],
            "url": rec['url'],
            "remote_testing": rec['remote_testing'],
            "adaptive_irt": rec['adaptive_irt'],
            "duration": rec['assessment_length'],
            "test_type": rec['test_type'],
            "relevance_score": rec['relevance_score']
        })
    return formatted

def main():
    parser = argparse.ArgumentParser(description="SHL Assessment Recommender System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build embeddings command
    build_parser = subparsers.add_parser("build", help="Build assessment embeddings")
    build_parser.add_argument("--data-file", type=str, default="data/shl_assessments.csv",
                              help="Path to the assessments data CSV file")
    build_parser.add_argument("--collection-name", type=str, default="shl_assessments",
                              help="Name of the vector collection to create")
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    recommend_parser.add_argument("query", type=str, help="Natural language query or job description")
    recommend_parser.add_argument("--job-url", type=str, help="URL to job description (optional)")
    recommend_parser.add_argument("--remote-testing", type=str, choices=["Yes", "No"],
                                help="Filter by remote testing support (Yes/No)")
    recommend_parser.add_argument("--adaptive-irt", type=str, choices=["Yes", "No"],
                                help="Filter by adaptive/IRT support (Yes/No)")
    recommend_parser.add_argument("--test-types", type=str,
                                help="Comma-separated list of test types to include")
    recommend_parser.add_argument("--limit", type=int, default=10,
                                help="Maximum number of recommendations to return")
    recommend_parser.add_argument("--collection-name", type=str, default="shl_assessments",
                                help="Name of the vector collection to use")
    recommend_parser.add_argument("--enhanced", action="store_true",
                                help="Use enhanced query processing with GPT")
    recommend_parser.add_argument("--output", type=str,
                                help="Output file to save recommendations (JSON)")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the recommender")
    evaluate_parser.add_argument("--queries-file", type=str, default="data/sample_test_queries.json",
                              help="JSON file containing test queries and relevant assessments")
    evaluate_parser.add_argument("--k", type=int, default=10,
                              help="Value of K for Recall@K and MAP@K metrics")
    evaluate_parser.add_argument("--collection-name", type=str, default="shl_assessments",
                              help="Name of the vector collection to use")
    evaluate_parser.add_argument("--enhanced", action="store_true",
                              help="Use enhanced query processing with GPT")
    evaluate_parser.add_argument("--output", type=str,
                              help="Output file to save detailed results (JSON)")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_embeddings(args.data_file, args.collection_name)
    
    elif args.command == "recommend":
        # Initialize vector store
        vector_store = QdrantVectorStore(collection_name=args.collection_name)
        
        # Parse test types if provided
        test_types = args.test_types.split(',') if args.test_types else None
        
        # Generate recommendations
        recommendations = recommend(
            vector_store,
            args.query,
            job_url=args.job_url,
            remote_testing=args.remote_testing,
            adaptive_irt=args.adaptive_irt,
            test_types=test_types,
            limit=args.limit,
            use_enhanced=args.enhanced
        )
        
        # Format and display recommendations
        formatted_recs = format_recommendations(recommendations)
        
        print(f"\nResults for query: '{args.query}'")
        print(f"Found {len(formatted_recs)} recommendations\n")
        
        for rec in formatted_recs:
            print(f"{rec['rank']}. {rec['name']}")
            print(f"   URL: {rec['url']}")
            print(f"   Remote Testing: {rec['remote_testing']}, Adaptive/IRT: {rec['adaptive_irt']}")
            print(f"   Duration: {rec['duration']}, Test Type: {rec['test_type']}")
            print(f"   Relevance Score: {rec['relevance_score']:.4f}\n")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(formatted_recs, f, indent=2)
            print(f"Recommendations saved to {args.output}")
    
    elif args.command == "evaluate":
        from evaluate_recommender import load_test_queries, evaluate_recommender
        
        # Initialize vector store and recommender
        vector_store = QdrantVectorStore(collection_name=args.collection_name)
        recommender = BaseRecommender(vector_store=vector_store)
        
        # Load test queries
        test_queries = load_test_queries(args.queries_file)
        print(f"Loaded {len(test_queries)} test queries")
        
        # Evaluate the recommender
        results = evaluate_recommender(
            recommender,
            test_queries,
            k=args.k,
            use_enhanced=args.enhanced
        )
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Mean Recall@{args.k}: {results['mean_recall_at_k']:.4f}")
        print(f"Mean MAP@{args.k}: {results['mean_map_at_k']:.4f}")
        
        # Save detailed results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to {args.output}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 