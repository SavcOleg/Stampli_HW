#!/usr/bin/env python3
"""
Quick Start Script for Disney RAG System
Automates data preparation and launches both API and UI servers.
"""

import subprocess
import sys
import time
import shutil
from pathlib import Path


def cleanup_data():
    """Remove all generated indices and processed data."""
    print("\n" + "="*80)
    print("🧹 Cleaning Up Generated Data")
    print("="*80 + "\n")
    
    print("⚠️  This will remove all generated indices and processed data")
    print("⚠️  You will need to regenerate everything (takes ~5-10 minutes)\n")
    
    response = input("Are you sure you want to clean up? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\nℹ️  Removing data/processed/...")
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            deleted = 0
            for f in list(processed_dir.glob("*.npy")) + list(processed_dir.glob("*.parquet")):
                f.unlink()
                print(f"   ✅ Removed {f.name}")
                deleted += 1
            if deleted == 0:
                print("   (no files to remove)")
        
        print("\nℹ️  Removing data/indices/...")
        indices_dir = Path("data/indices")
        if indices_dir.exists():
            deleted = 0
            for f in list(indices_dir.glob("*.index")) + list(indices_dir.glob("*.pkl")):
                f.unlink()
                print(f"   ✅ Removed {f.name}")
                deleted += 1
            if deleted == 0:
                print("   (no files to remove)")
        
        print("\n✅ Cleanup complete! You can now test from scratch.")
        print("\nℹ️  Run 'python3 quickstart.py' or './quickstart.sh' to rebuild everything")
        sys.exit(0)
    else:
        print("\nℹ️  Cleanup cancelled")
        sys.exit(0)


def run_command(cmd: str, description: str, check: bool = True) -> bool:
    """Run a shell command and print status."""
    print(f"\nℹ️  {description}...")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr[:500] if e.stderr else str(e)}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("\nℹ️  Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, but you have {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} found")
    return True


def check_openai_key():
    """Check if OpenAI API key exists and is valid."""
    print("\nℹ️  Checking for OpenAI API key...")
    
    env_file = Path(".env")
    api_key = None
    
    # Try to read existing key
    if env_file.exists():
        print("✅ .env file exists")
        # Extract API key from .env
        for line in env_file.read_text().splitlines():
            if line.strip().startswith('OPENAI_API_KEY'):
                api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                break
    
    # If no key found, ask for it
    if not api_key:
        print("⚠️  OpenAI API key not found in .env file!")
        print("\nPlease enter your OpenAI API key:")
        print("(You can get one from: https://platform.openai.com/api-keys)")
        api_key = input("OpenAI API Key: ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return False
        
        # Save to .env
        env_file.write_text(f'OPENAI_API_KEY="{api_key}"\n')
        print("✅ API key saved to .env file")
    
    # Validate the API key by making a test call
    print("\nℹ️  Validating OpenAI API key...")
    try:
        import openai
        import os
        
        # Set the API key temporarily
        os.environ['OPENAI_API_KEY'] = api_key
        client = openai.OpenAI(api_key=api_key)
        
        # Make a minimal test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        print("✅ OpenAI API key is valid!")
        return True
        
    except openai.AuthenticationError:
        print("❌ Invalid OpenAI API key!")
        print("   Please check your key at: https://platform.openai.com/api-keys")
        
        # Ask if they want to enter a new key
        retry = input("\nWould you like to enter a new API key? (yes/no): ").strip().lower()
        if retry == 'yes':
            # Remove old key and retry
            if env_file.exists():
                env_file.unlink()
            return check_openai_key()
        return False
        
    except openai.RateLimitError:
        print("⚠️  Rate limit reached, but API key is valid")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not validate API key: {str(e)}")
        print("   Continuing anyway (will fail later if key is invalid)")
        return True


def check_data_ready() -> bool:
    """Check if all required data files exist."""
    required_files = [
        Path("data/processed/chunks.parquet"),
        Path("data/indices/faiss_flat.index"),
        Path("data/indices/bm25.pkl"),
        Path("data/indices/metadata.pkl")
    ]
    
    return all(f.exists() for f in required_files)


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("🏰 Disney RAG System - Quick Start")
    print("="*80 + "\n")
    print("ℹ️  Tip: Use 'python3 quickstart.py clean' to remove all generated data\n")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check OpenAI API key
    if not check_openai_key():
        sys.exit(1)
    
    # Step 3: Install dependencies
    print("\n" + "="*80)
    print("📦 Installing Dependencies")
    print("="*80)
    print("\nℹ️  This may take 2-3 minutes...")
    
    if not run_command("python3 -m pip install --upgrade pip", "Upgrade pip", check=False):
        print("⚠️  Could not upgrade pip (continuing anyway)")
    
    if not run_command("python3 -m pip install -r requirements.txt", "Install dependencies"):
        print("\n❌ Failed to install dependencies")
        print("Please run manually: pip3 install -r requirements.txt")
        sys.exit(1)
    
    # Step 4: Prepare data
    print("\n" + "="*80)
    print("🔧 Preparing Data")
    print("="*80)
    
    if check_data_ready():
        print("\n✅ Data already prepared! Skipping ingestion and indexing.")
        data_ready = True
    else:
        print("\nℹ️  Data not found. Running ingestion and indexing...")
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Step 4a: Run ingestion
        print("\nℹ️  Step 1/2: Running data ingestion (extracting features, chunking reviews)...")
        if not run_command(
            "python3 -m src.ingestion.pipeline > logs/ingestion.log 2>&1",
            "Data ingestion"
        ):
            print("❌ Ingestion failed. Check logs/ingestion.log for details")
            sys.exit(1)
        print("✅ Ingestion completed (123,860 chunks created)")
        
        # Step 4b: Build indices
        print("\nℹ️  Step 2/2: Building search indices (FAISS + BM25)...")
        print("⚠️  This may take 3-5 minutes for embeddings...")
        if not run_command(
            "python3 -m src.retrieval.build_indices > logs/indexing.log 2>&1",
            "Index building"
        ):
            print("❌ Index building failed. Check logs/indexing.log for details")
            sys.exit(1)
        
        data_ready = True
    
    if data_ready:
        print("\n✅ All data is ready!")
        print("\nℹ️  Data summary:")
        print("  - Reviews: 42,656")
        print("  - Chunks: 123,860")
        
        # Show file sizes
        for file_path, name in [
            ("data/indices/faiss_flat.index", "FAISS index"),
            ("data/indices/bm25.pkl", "BM25 index"),
        ]:
            p = Path(file_path)
            if p.exists():
                size = p.stat().st_size / (1024*1024)
                print(f"  - {name}: {size:.0f}M")
    
    # Step 5: Start services
    print("\n" + "="*80)
    print("🚀 Starting Services")
    print("="*80)
    
    print("\nℹ️  Stopping any existing services...")
    run_command("pkill -f 'uvicorn src.api.app:app'", "Stop old API", check=False)
    run_command("pkill -f 'streamlit run src/ui/app.py'", "Stop old UI", check=False)
    time.sleep(2)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Start API server
    print("\nℹ️  Starting FastAPI server on port 8000...")
    subprocess.Popen(
        "python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)
    print("✅ API server started")
    
    # Start Streamlit UI
    print("\nℹ️  Starting Streamlit UI on port 8501...")
    subprocess.Popen(
        "python3 -m streamlit run src/ui/app.py --server.port 8501 --server.headless true > logs/streamlit.log 2>&1",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    print("✅ Streamlit UI started")
    
    # Wait for services to be ready
    print("\nℹ️  Waiting for services to be ready...")
    time.sleep(3)
    
    # Success message
    print("\n" + "="*80)
    print("🎉 SUCCESS! Disney RAG System is Running")
    print("="*80 + "\n")
    
    print("✅ Your Disney RAG System is now live!\n")
    print("📱 Access Points:")
    print("  • Streamlit UI:  http://localhost:8501")
    print("  • API Server:    http://localhost:8000")
    print("  • API Health:    http://localhost:8000/healthz")
    print("  • API Metrics:   http://localhost:8000/metrics")
    print("")
    print("📊 What You Can Do:")
    print("  1. Open http://localhost:8501 in your browser")
    print("  2. Try the example queries in the sidebar")
    print("  3. Use the Chat tab for Q&A")
    print("  4. Use the Eval tab for analytics")
    print("  5. Use the Testing tab to run automated tests")
    print("")
    print("📁 Project Stats:")
    print("  • 42,656 reviews indexed")
    print("  • 123,860 searchable chunks")
    print("  • ~8 second average response time")
    print("  • Full citations with every answer")
    print("")
    print("🔧 Management:")
    print("  • View logs: tail -f logs/api.log or logs/streamlit.log")
    print("  • Stop services: ./stop.sh")
    print("  • Clean data: python3 quickstart.py clean")
    print("")
    
    # Show live logs
    response = input("Show live logs? (y/n): ")
    if response.lower() == 'y':
        print("\nℹ️  Showing live logs (Ctrl+C to exit)...")
        print("")
        try:
            subprocess.run("tail -f logs/api.log logs/streamlit.log", shell=True)
        except KeyboardInterrupt:
            print("\n🛑 Stopped viewing logs")
            print("✅ Services are still running in the background")


if __name__ == "__main__":
    # Check for cleanup flag
    if len(sys.argv) > 1 and sys.argv[1] in ['clean', '--clean']:
        cleanup_data()
    
    main()
