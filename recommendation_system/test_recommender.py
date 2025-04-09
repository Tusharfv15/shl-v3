import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent))

from utils.vector_store import QdrantVectorStore
from models.recommender import BaseRecommender
from main import SHLRecommender

def format_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    """
    Format and print recommendations
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    print("\nRecommendations:\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   URL: {rec['url']}")
        print(f"   Remote Testing: {rec['remote_testing']}, Adaptive/IRT: {rec['adaptive_irt']}")
        print(f"   Assessment Length: {rec['assessment_length']}")
        print(f"   Test Type: {rec['test_type']}")
        print(f"   Relevance Score: {rec.get('relevance_score', 'N/A'):.4f}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Test SHL Assessment Recommender")
    parser.add_argument("query", type=str, help="Natural language query or job description")
    parser.add_argument("--job-url", type=str, help="URL to job description (optional)")
    parser.add_argument("--remote-testing", type=str, choices=["Yes", "No"],
                      help="Filter by remote testing support (Yes/No)")
    parser.add_argument("--adaptive-irt", type=str, choices=["Yes", "No"],
                      help="Filter by adaptive/IRT support (Yes/No)")
    parser.add_argument("--test-types", type=str,
                      help="Comma-separated list of test types to include")
    parser.add_argument("--limit", type=int, default=5,
                      help="Maximum number of recommendations to return")
    parser.add_argument("--collection-name", type=str, default="shl_assessments",
                      help="Name of the vector collection to use")
    parser.add_argument("--enhanced", action="store_true",
                      help="Use enhanced query processing with GPT")
    parser.add_argument("--output", type=str,
                      help="Output file to save recommendations (JSON)")
    parser.add_argument("--use-wrapper", action="store_true",
                      help="Use the SHLRecommender wrapper class instead of direct BaseRecommender")
    
    args = parser.parse_args()
    
    # Parse test types if provided
    test_types = None
    if args.test_types:
        test_types = [t.strip() for t in args.test_types.split(',')]
    
    # Create filter dictionary
    filters = {}
    if args.remote_testing:
        filters["remote_testing"] = args.remote_testing
    if args.adaptive_irt:
        filters["adaptive_irt"] = args.adaptive_irt
    if test_types:
        filters["test_type"] = test_types
    
    print(f"Query: {args.query}")
    print(f"Enhanced mode: {'Yes' if args.enhanced else 'No'}")
    if filters:
        print("Filters:")
        for key, value in filters.items():
            print(f"  {key}: {value}")
    
    # Initialize recommender
    if args.use_wrapper:
        print(f"Initializing SHLRecommender with collection '{args.collection_name}'...")
        recommender = SHLRecommender(collection_name=args.collection_name)
        
        # Get recommendations
        if args.job_url:
            print(f"Getting recommendations from URL: {args.job_url}...")
            recommendations = recommender.recommend_from_url(
                url=args.job_url,
                top_k=args.limit,
                enhanced=args.enhanced,
                filters=filters if filters else None
            )
        else:
            print("Getting recommendations...")
            recommendations = recommender.recommend(
                query=args.query,
                top_k=args.limit,
                enhanced=args.enhanced,
                filters=filters if filters else None
            )
    else:
        # Use the BaseRecommender directly
        print(f"Initializing BaseRecommender with collection '{args.collection_name}'...")
        vector_store = QdrantVectorStore(collection_name=args.collection_name)
        recommender = BaseRecommender(vector_store=vector_store)
        
        # Get recommendations
        if args.enhanced:
            print("Using enhanced mode with GPT...")
            recommendations = recommender.enhanced_recommendations(
                args.query,
                job_description_url=args.job_url,
                remote_testing=args.remote_testing,
                adaptive_irt=args.adaptive_irt,
                test_types=test_types,
                limit=args.limit
            )
        else:
            print("Using basic mode...")
            recommendations = recommender.process_query(
                args.query,
                remote_testing=args.remote_testing,
                adaptive_irt=args.adaptive_irt,
                test_types=test_types,
                limit=args.limit
            )
    
    # Format and display recommendations
    print(f"Found {len(recommendations)} recommendations")
    format_recommendations(recommendations)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Recommendations saved to {args.output}")

if __name__ == "__main__":
    main() 