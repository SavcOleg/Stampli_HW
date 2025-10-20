"""Quick test script for API endpoints."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*80)
    print("Testing /healthz endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/healthz")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_query():
    """Test query endpoint."""
    print("\n" + "="*80)
    print("Testing /query endpoint")
    print("="*80)
    
    payload = {
        "query": "Is the staff friendly?",
        "top_k": 5,
        "enable_filtering": True,
        "return_contexts": False
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/query", json=payload)
    latency = (time.time() - start) * 1000
    
    print(f"\nStatus: {response.status_code}")
    print(f"Latency: {latency:.0f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Answer:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        print(f"\n📊 Metrics:")
        print(f"   Citations: {len(result['citations'])}")
        print(f"   Contexts: {result['metrics']['num_contexts']}")
        print(f"   Retrieval: {result['metrics']['retrieval_latency_ms']:.0f}ms")
        print(f"   Generation: {result['metrics']['generation_latency_ms']:.0f}ms")
        print(f"   Total: {result['metrics']['total_latency_ms']:.0f}ms")
        return True
    else:
        print(f"\n❌ Error: {response.text}")
        return False

def test_metrics():
    """Test metrics endpoint."""
    print("\n" + "="*80)
    print("Testing /metrics endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        lines = response.text.split('\n')[:20]
        print("First 20 lines of metrics:")
        for line in lines:
            if line and not line.startswith('#'):
                print(f"  {line}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 API Test Suite")
    print("="*80)
    print("\nMake sure the API server is running:")
    print("  uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
    print("\nPress Enter to start tests...")
    input()
    
    # Run tests
    results = {
        "health": test_health(),
        "query": test_query(),
        "metrics": test_metrics()
    }
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))

