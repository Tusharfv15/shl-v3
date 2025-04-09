import json
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Union
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path so we can import our modules
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from utils.vector_store import QdrantVectorStore
from models.recommender import BaseRecommender

def load_test_queries(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test queries from a JSON file
    
    Args:
        file_path: Path to the JSON file containing test queries
        
    Returns:
        List of test query dictionaries
    """
    with open(file_path, 'r') as f:
        queries = json.load(f)
    return queries

def calculate_recall_at_k(relevant: List[str], retrieved: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate Recall@K
    
    Args:
        relevant: List of relevant assessment names
        retrieved: List of retrieved assessment dictionaries
        k: The K value for Recall@K
        
    Returns:
        Recall@K score
    """
    if not relevant:
        return 0.0
    
    # Get names of retrieved assessments
    retrieved_names = [item['name'] for item in retrieved[:k]]
    
    # Count relevant items found
    found = sum(1 for name in relevant if name in retrieved_names)
    
    # Calculate recall
    return found / len(relevant)

def calculate_map_at_k(relevant: List[str], retrieved: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate Mean Average Precision at K (MAP@K)
    
    Args:
        relevant: List of relevant assessment names
        retrieved: List of retrieved assessment dictionaries
        k: The K value for MAP@K
        
    Returns:
        MAP@K score
    """
    if not relevant:
        return 0.0
    
    # Get names of retrieved assessments
    retrieved_names = [item['name'] for item in retrieved[:k]]
    
    # Calculate precision at each relevant item
    precision_sum = 0.0
    num_hits = 0
    
    for i, name in enumerate(retrieved_names):
        if name in relevant:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            precision_sum += precision_at_i
    
    # Calculate average precision
    if num_hits == 0:
        return 0.0
    
    return precision_sum / min(len(relevant), k)

def evaluate_recommender(recommender: BaseRecommender, test_queries: List[Dict[str, Any]], 
                        k: int = 10, use_enhanced: bool = False) -> Dict[str, Any]:
    """
    Evaluate the recommender using the provided test queries
    
    Args:
        recommender: The recommender to evaluate
        test_queries: List of test query dictionaries
        k: The K value for Recall@K and MAP@K
        use_enhanced: Whether to use enhanced query processing
        
    Returns:
        Dictionary of evaluation results
    """
    results = {
        'queries': [],
        'mean_recall_at_k': 0.0,
        'mean_map_at_k': 0.0
    }
    
    recall_scores = []
    map_scores = []
    
    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query = query_data['query']
        relevant_assessments = query_data['relevant_assessments']
        
        # Get recommendations
        if use_enhanced:
            recommendations = recommender.enhanced_recommendations(query, limit=k)
        else:
            recommendations = recommender.process_query(query, limit=k)
        
        # Calculate metrics
        recall = calculate_recall_at_k(relevant_assessments, recommendations, k)
        map_score = calculate_map_at_k(relevant_assessments, recommendations, k)
        
        # Save individual query results
        query_result = {
            'query': query,
            'description': query_data.get('description', ''),
            'relevant_assessments': relevant_assessments,
            'recommendations': [rec['name'] for rec in recommendations[:k]],
            'recall_at_k': recall,
            'map_at_k': map_score
        }
        
        results['queries'].append(query_result)
        recall_scores.append(recall)
        map_scores.append(map_score)
    
    # Calculate mean scores
    results['mean_recall_at_k'] = np.mean(recall_scores) if recall_scores else 0.0
    results['mean_map_at_k'] = np.mean(map_scores) if map_scores else 0.0
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL Assessment Recommender")
    parser.add_argument("--queries-file", type=str, default="data/sample_test_queries.json",
                      help="JSON file containing test queries and relevant assessments")
    parser.add_argument("--k", type=int, default=10,
                      help="Value of K for Recall@K and MAP@K metrics")
    parser.add_argument("--collection-name", type=str, default="shl_assessments",
                      help="Name of the vector collection to use")
    parser.add_argument("--enhanced", action="store_true",
                      help="Use enhanced query processing with GPT")
    parser.add_argument("--output", type=str,
                      help="Output file to save detailed results (JSON)")
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main() 