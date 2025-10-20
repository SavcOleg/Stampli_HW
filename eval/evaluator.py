"""
Evaluation framework for RAG system.

Metrics:
- Retrieval quality (relevance, coverage)
- Generation quality (citations, groundedness)
- System performance (latency, tokens)
"""

import sys
from pathlib import Path
import yaml
import time
from typing import Dict, List
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.generator import RAGGenerator


class RAGEvaluator:
    """Evaluate RAG system on gold dataset."""
    
    def __init__(self, gold_dataset_path: str = "eval/gold_dataset.yaml"):
        """
        Initialize evaluator.
        
        Args:
            gold_dataset_path: Path to gold dataset YAML
        """
        self.gold_dataset_path = Path(gold_dataset_path)
        self.load_gold_dataset()
        self.generator = None
    
    def load_gold_dataset(self):
        """Load gold dataset from YAML."""
        with open(self.gold_dataset_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.queries = data['queries']
        self.thresholds = data['thresholds']
        
        print(f"✅ Loaded {len(self.queries)} gold queries")
    
    def initialize_generator(self):
        """Initialize RAG generator."""
        if not self.generator:
            print("Initializing RAG generator...")
            self.generator = RAGGenerator()
    
    def evaluate_query(self, query_data: Dict) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            query_data: Query data from gold dataset
        
        Returns:
            Evaluation results
        """
        query = query_data['question']
        
        # Generate answer
        start_time = time.time()
        result = self.generator.generate(query=query, return_contexts=True)
        eval_latency_ms = (time.time() - start_time) * 1000
        
        # Extract metrics
        metrics = result['metrics']
        citations = result['citations']
        filters = result['filters_applied']
        contexts = result.get('contexts', [])
        
        # Evaluation results
        eval_result = {
            'query': query,
            'category': query_data.get('category', 'unknown'),
            'answer_length': len(result['answer']),
            'num_citations': len(citations),
            'total_latency_ms': metrics['total_latency_ms'],
            'retrieval_latency_ms': metrics['retrieval_latency_ms'],
            'generation_latency_ms': metrics['generation_latency_ms'],
            'total_tokens': metrics['total_tokens'],
            'num_contexts': len(contexts),
            'filters_applied': filters,
        }
        
        # Check thresholds
        eval_result['meets_latency_threshold'] = (
            metrics['total_latency_ms'] <= self.thresholds['max_latency_ms']
        )
        eval_result['meets_citation_threshold'] = (
            len(citations) >= query_data.get('min_citations', self.thresholds['min_citations'])
        )
        eval_result['meets_token_threshold'] = (
            metrics['total_tokens'] <= self.thresholds['max_tokens']
        )
        
        # Check expected filters
        if 'expected_parks' in query_data:
            expected_park = query_data['expected_parks'][0].lower()
            actual_park = filters.get('park', '').lower()
            eval_result['correct_park_filter'] = (expected_park == actual_park)
        
        if 'expected_countries' in query_data:
            expected_country = query_data['expected_countries'][0].lower()
            actual_country = filters.get('country', '').lower()
            eval_result['correct_country_filter'] = (expected_country == actual_country)
        
        if 'expected_seasons' in query_data:
            expected_season = query_data['expected_seasons'][0].lower()
            actual_season = filters.get('season', '').lower()
            eval_result['correct_season_filter'] = (expected_season == actual_season)
        
        # Overall pass/fail
        eval_result['passed'] = (
            eval_result['meets_latency_threshold'] and
            eval_result['meets_citation_threshold'] and
            eval_result['meets_token_threshold']
        )
        
        return eval_result
    
    def run_evaluation(self) -> pd.DataFrame:
        """
        Run evaluation on all gold queries.
        
        Returns:
            DataFrame with evaluation results
        """
        self.initialize_generator()
        
        print("\n" + "="*80)
        print("🧪 Running RAG Evaluation")
        print("="*80 + "\n")
        
        results = []
        
        for i, query_data in enumerate(self.queries, 1):
            print(f"Evaluating query {i}/{len(self.queries)}: {query_data['question'][:60]}...")
            
            try:
                eval_result = self.evaluate_query(query_data)
                results.append(eval_result)
                
                status = "✅ PASS" if eval_result['passed'] else "⚠️  FAIL"
                print(f"  {status} - Latency: {eval_result['total_latency_ms']:.0f}ms, "
                      f"Citations: {eval_result['num_citations']}, "
                      f"Tokens: {eval_result['total_tokens']}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                results.append({
                    'query': query_data['question'],
                    'category': query_data.get('category'),
                    'error': str(e),
                    'passed': False
                })
        
        df = pd.DataFrame(results)
        return df
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate evaluation report.
        
        Args:
            df: DataFrame with evaluation results
        
        Returns:
            Report string
        """
        report = []
        report.append("\n" + "="*80)
        report.append("📊 EVALUATION REPORT")
        report.append("="*80 + "\n")
        
        # Overall statistics
        total = len(df)
        passed = df['passed'].sum()
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        report.append(f"**Overall Results**")
        report.append(f"  Total Queries: {total}")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {total - passed}")
        report.append(f"  Pass Rate: {pass_rate:.1f}%")
        report.append("")
        
        # Performance metrics
        if 'total_latency_ms' in df.columns:
            report.append(f"**Performance Metrics**")
            report.append(f"  Avg Total Latency: {df['total_latency_ms'].mean():.0f}ms")
            report.append(f"  Avg Retrieval: {df['retrieval_latency_ms'].mean():.0f}ms")
            report.append(f"  Avg Generation: {df['generation_latency_ms'].mean():.0f}ms")
            report.append(f"  Avg Tokens: {df['total_tokens'].mean():.0f}")
            report.append(f"  Avg Citations: {df['num_citations'].mean():.1f}")
            report.append("")
        
        # Threshold compliance
        if 'meets_latency_threshold' in df.columns:
            report.append(f"**Threshold Compliance**")
            report.append(f"  Latency (<{self.thresholds['max_latency_ms']}ms): "
                         f"{df['meets_latency_threshold'].sum()}/{total}")
            report.append(f"  Citations (≥{self.thresholds['min_citations']}): "
                         f"{df['meets_citation_threshold'].sum()}/{total}")
            report.append(f"  Tokens (<{self.thresholds['max_tokens']}): "
                         f"{df['meets_token_threshold'].sum()}/{total}")
            report.append("")
        
        # Category breakdown
        if 'category' in df.columns:
            report.append(f"**Performance by Category**")
            for category, group in df.groupby('category'):
                cat_pass_rate = (group['passed'].sum() / len(group) * 100)
                report.append(f"  {category}: {group['passed'].sum()}/{len(group)} ({cat_pass_rate:.0f}%)")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, df: pd.DataFrame, output_path: str = "logs/eval_results.csv"):
        """Save results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to {output_path}")


def main():
    """Main evaluation entry point."""
    evaluator = RAGEvaluator()
    
    # Run evaluation
    df = evaluator.run_evaluation()
    
    # Generate report
    report = evaluator.generate_report(df)
    print(report)
    
    # Save results
    evaluator.save_results(df)
    
    # Save report
    report_path = Path("logs/eval_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()

