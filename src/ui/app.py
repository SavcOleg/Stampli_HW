"""
Streamlit UI for Disney RAG System.

Features:
- Chat tab: Interactive Q&A with citations
- Eval tab: Evaluation metrics and visualizations
"""

import streamlit as st
import sys
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.generator import RAGGenerator

# Page config
st.set_page_config(
    page_title="Disney RAG System",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f7f7f;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_generator():
    """Load RAG generator (cached)."""
    with st.spinner("🔄 Initializing RAG system..."):
        generator = RAGGenerator(top_k=10)
    return generator


def chat_tab():
    """Chat interface for interactive queries."""
    st.markdown('<p class="main-header">🏰 Disney RAG Chat</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about Disney parks based on visitor reviews</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Number of chunks", 
            min_value=3, 
            max_value=20, 
            value=10,
            help="How many review chunks to retrieve. More chunks = more context but slower response."
        )
        enable_filtering = st.checkbox(
            "Enable metadata filtering", 
            value=True,
            help="Automatically filter by park, country, season based on your query (e.g., 'Hong Kong in spring')"
        )
        show_contexts = st.checkbox(
            "Show retrieved contexts", 
            value=False,
            help="Display the actual review chunks used to generate the answer (useful for debugging)"
        )
        
        st.divider()
        
        st.header("📚 Example Queries")
        example_queries = [
            "What do visitors from Australia say about Disneyland in Hong Kong?",
            "Is spring a good time to visit Disneyland?",
            "Is Disneyland California crowded in June?",
            "Is the staff in Paris friendly?",
            "What are the best rides for families with young children?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query[:30]}"):
                st.session_state.current_query = query
        
        st.divider()
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Query input
    st.info("💡 **Tip**: Ask about specific parks (Hong Kong, Paris, California), seasons (spring, summer), or topics (staff, food, rides)")
    
    query = st.text_input(
        "Ask a question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., Is the staff friendly?",
        key="query_input",
        help="Type your question about Disney parks. The system will search 123K+ review chunks and provide an answer with citations."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Search", type="primary")
    with col2:
        if st.session_state.chat_history:
            st.caption(f"💬 {len(st.session_state.chat_history)} questions asked")
    
    # Process query
    if submit_button and query:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            try:
                generator = load_generator()
                result = generator.generate(
                    query=query,
                    enable_filtering=enable_filtering,
                    return_contexts=show_contexts
                )
                
                # Add to history
                st.session_state.chat_history.append({
                    'query': query,
                    'result': result,
                    'timestamp': time.time()
                })
                
                st.success(f"✅ Completed in {result['metrics']['total_latency_ms']/1000:.2f}s")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
    
    # Display chat history (most recent first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.divider()
        
        # Query
        st.markdown(f"### 💬 Query {len(st.session_state.chat_history) - i}")
        st.markdown(f"**{chat['query']}**")
        
        result = chat['result']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "⏱️ Total Time", 
                f"{result['metrics']['total_latency_ms']:.0f}ms",
                help="Total response time: retrieval (~350ms) + generation (~6-8s)"
            )
        with col2:
            st.metric(
                "📚 Citations", 
                len(result['citations']),
                help="Number of unique review IDs cited in the answer"
            )
        with col3:
            st.metric(
                "📄 Contexts", 
                result['metrics']['num_contexts'],
                help="Number of review chunks retrieved and used for context"
            )
        with col4:
            st.metric(
                "🔤 Tokens", 
                result['metrics']['total_tokens'],
                help="Total tokens used (prompt + completion) - affects API cost (~$0.0002/query)"
            )
        
        # Answer
        st.markdown("### 💡 Answer")
        st.markdown(result['answer'])
        
        # Citations
        if result['citations']:
            with st.expander(f"📚 Citations ({len(result['citations'])})"):
                st.caption("These review IDs were cited in the answer above. All claims are grounded in actual visitor reviews.")
                for citation in result['citations']:
                    st.markdown(f'<div class="citation">Review ID: {citation}</div>', unsafe_allow_html=True)
        
        # Filters applied
        if result['filters_applied']:
            with st.expander("🔍 Filters Applied"):
                st.caption("The system automatically detected these filters from your query to narrow down results.")
                st.json(result['filters_applied'])
        elif enable_filtering:
            st.caption("💡 No filters detected. Try mentioning a park name, country, or season in your query!")
        
        
        # Contexts
        if show_contexts and result.get('contexts'):
            with st.expander(f"📄 Retrieved Contexts ({len(result['contexts'])})"):
                st.caption("These are the actual review chunks retrieved and ranked by relevance. The LLM uses these to generate the answer.")
                for j, ctx in enumerate(result['contexts'], 1):
                    st.markdown(f"**{j}. Park: {ctx['park']}, Country: {ctx['country']}, Rating: {ctx['rating']}★**")
                    st.text(ctx['chunk_text'][:200] + "...")
                    st.caption(f"Hybrid Score: {ctx.get('score', 0):.4f} | Re-rank Score: {ctx.get('rerank_score', 'N/A')}")
        
        # Feedback
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{i}"):
                st.success("Thanks for feedback!")
        with col2:
            if st.button("👎 Not helpful", key=f"not_helpful_{i}"):
                st.info("Thanks for feedback!")


def testing_tab():
    """Testing and validation interface."""
    st.markdown('<p class="main-header">🧪 Testing & Validation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Run tests on gold dataset queries and validate system performance</p>', unsafe_allow_html=True)
    
    # Load gold dataset
    try:
        import yaml
        gold_path = Path("eval/gold_dataset.yaml")
        
        if not gold_path.exists():
            st.error("❌ Gold dataset not found at eval/gold_dataset.yaml")
            return
        
        with open(gold_path, 'r') as f:
            gold_data = yaml.safe_load(f)
        
        queries = gold_data['queries']
        thresholds = gold_data['thresholds']
        
        st.success(f"✅ Loaded {len(queries)} test queries from gold dataset")
        
        # Display thresholds
        with st.expander("📏 Quality Thresholds"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Latency", f"{thresholds['max_latency_ms']}ms", help="Maximum acceptable response time")
            with col2:
                st.metric("Min Citations", thresholds['min_citations'], help="Minimum citations required per answer")
            with col3:
                st.metric("Max Tokens", thresholds['max_tokens'], help="Maximum tokens per query (cost control)")
            with col4:
                st.metric("Min Recall", f"{thresholds.get('min_retrieval_recall', 0.6)*100:.0f}%", help="Minimum retrieval recall expected")
        
        st.divider()
        
        # Test queries table
        st.markdown("### 📋 Gold Dataset Test Queries")
        st.caption("These queries represent key use cases and are used for automated evaluation")
        
        # Prepare table data
        table_data = []
        for i, q in enumerate(queries, 1):
            table_data.append({
                '#': i,
                'Question': q['question'],
                'Category': q.get('category', 'general'),
                'Expected Parks': ', '.join(q.get('expected_parks', [])) or '-',
                'Expected Countries': ', '.join(q.get('expected_countries', [])) or '-',
                'Expected Seasons': ', '.join(q.get('expected_seasons', [])) or '-',
                'Min Citations': q.get('min_citations', thresholds['min_citations'])
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        st.divider()
        
        # Test runner section
        st.markdown("### 🚀 Run Tests")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            test_mode = st.radio(
                "Test Mode:",
                ["Single Query", "All Queries"],
                help="Run a single test query or all queries in the gold dataset"
            )
        
        with col2:
            if test_mode == "Single Query":
                selected_idx = st.selectbox(
                    "Select Query:",
                    range(len(queries)),
                    format_func=lambda x: f"Q{x+1}: {queries[x]['question'][:50]}...",
                    help="Choose which query to test"
                )
            else:
                st.info(f"Will run all {len(queries)} queries (~{len(queries)*8} seconds)")
        
        with col3:
            run_button = st.button("▶️ Run Test", type="primary", use_container_width=True)
        
        # Run tests
        if run_button:
            if 'test_results' not in st.session_state:
                st.session_state.test_results = []
            
            # Determine which queries to run
            if test_mode == "Single Query":
                queries_to_run = [queries[selected_idx]]
                query_indices = [selected_idx]
            else:
                queries_to_run = queries
                query_indices = list(range(len(queries)))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load generator
            generator = load_generator()
            
            results = []
            for idx, (i, query_data) in enumerate(zip(query_indices, queries_to_run)):
                query = query_data['question']
                
                status_text.text(f"Testing query {idx+1}/{len(queries_to_run)}: {query[:60]}...")
                progress_bar.progress((idx + 1) / len(queries_to_run))
                
                # Run query
                try:
                    result = generator.generate(query=query, return_contexts=False)
                    
                    # Evaluate
                    metrics = result['metrics']
                    citations = result['citations']
                    filters = result['filters_applied']
                    
                    # Check thresholds
                    passes_latency = metrics['total_latency_ms'] <= thresholds['max_latency_ms']
                    passes_citations = len(citations) >= query_data.get('min_citations', thresholds['min_citations'])
                    passes_tokens = metrics['total_tokens'] <= thresholds['max_tokens']
                    
                    # Check expected filters
                    correct_filters = {}
                    if 'expected_parks' in query_data:
                        expected = query_data['expected_parks'][0].lower()
                        actual = filters.get('park', '').lower()
                        correct_filters['park'] = (expected == actual)
                    
                    if 'expected_countries' in query_data:
                        expected = query_data['expected_countries'][0].lower()
                        actual = filters.get('country', '').lower()
                        correct_filters['country'] = (expected == actual)
                    
                    if 'expected_seasons' in query_data:
                        expected = query_data['expected_seasons'][0].lower()
                        actual = filters.get('season', '').lower()
                        correct_filters['season'] = (expected == actual)
                    
                    filters_correct = all(correct_filters.values()) if correct_filters else None
                    
                    # Overall pass
                    passed = passes_latency and passes_citations and passes_tokens
                    if filters_correct is not None:
                        passed = passed and filters_correct
                    
                    results.append({
                        'Query #': i + 1,
                        'Question': query[:60] + '...' if len(query) > 60 else query,
                        'Category': query_data.get('category', 'general'),
                        'Status': '✅ PASS' if passed else '⚠️ FAIL',
                        'Latency (ms)': f"{metrics['total_latency_ms']:.0f}",
                        'Latency OK': '✅' if passes_latency else '❌',
                        'Citations': len(citations),
                        'Citations OK': '✅' if passes_citations else '❌',
                        'Tokens': metrics['total_tokens'],
                        'Tokens OK': '✅' if passes_tokens else '❌',
                        'Filters OK': '✅' if filters_correct else ('❌' if filters_correct is False else '-'),
                    })
                    
                except Exception as e:
                    results.append({
                        'Query #': i + 1,
                        'Question': query[:60] + '...',
                        'Category': query_data.get('category', 'general'),
                        'Status': '❌ ERROR',
                        'Latency (ms)': '-',
                        'Latency OK': '❌',
                        'Citations': 0,
                        'Citations OK': '❌',
                        'Tokens': 0,
                        'Tokens OK': '❌',
                        'Filters OK': '❌',
                    })
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.test_results = results
            
            st.success(f"✅ Completed {len(results)} test(s)")
        
        # Display results
        if 'test_results' in st.session_state and st.session_state.test_results:
            st.divider()
            st.markdown("### 📊 Test Results")
            
            results_df = pd.DataFrame(st.session_state.test_results)
            
            # Summary metrics
            total = len(results_df)
            passed = len(results_df[results_df['Status'] == '✅ PASS'])
            failed = total - passed
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tests", total)
            with col2:
                st.metric("Passed", passed, delta=f"{pass_rate:.1f}%")
            with col3:
                st.metric("Failed", failed, delta=f"{-100+pass_rate:.1f}%" if failed > 0 else None)
            with col4:
                status_color = "🟢" if pass_rate >= 80 else "🟡" if pass_rate >= 60 else "🔴"
                st.metric("Status", f"{status_color} {pass_rate:.0f}%")
            
            st.divider()
            
            # Results table with color coding
            st.markdown("#### Detailed Results")
            st.dataframe(
                results_df.style.apply(
                    lambda x: ['background-color: #d4edda' if v == '✅ PASS' 
                              else 'background-color: #f8d7da' if v in ['⚠️ FAIL', '❌ ERROR']
                              else '' for v in x],
                    subset=['Status']
                ),
                use_container_width=True,
                height=400
            )
            
            # Export results
            col1, col2 = st.columns([3, 1])
            with col2:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="test_results.csv",
                    mime="text/csv",
                    help="Download test results as CSV file"
                )
        
    except ImportError:
        st.error("❌ PyYAML not installed. Run: pip install pyyaml")
    except Exception as e:
        st.error(f"❌ Error loading gold dataset: {str(e)}")


def eval_tab():
    """Evaluation metrics and visualizations."""
    st.markdown('<p class="main-header">📊 Evaluation Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Performance analysis and system metrics</p>', unsafe_allow_html=True)
    
    # Check if we have chat history
    if not st.session_state.get('chat_history'):
        st.info("👈 **Run some queries in the Chat tab first to see evaluation metrics here!**")
        st.caption("The Eval tab provides analytics on your queries: latency breakdown, citation counts, token usage, and performance trends.")
        
        # Show sample metrics visualization
        st.markdown("### 📈 Sample Metrics (Demo Data)")
        st.caption("This is sample data to show what the charts will look like. Your real data will appear after running queries.")
        
        sample_data = {
            'Query': ['Query 1', 'Query 2', 'Query 3', 'Query 4'],
            'Retrieval (ms)': [850, 920, 780, 810],
            'Generation (ms)': [7500, 8200, 6900, 7800],
            'Total (ms)': [8350, 9120, 7680, 8610]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Latency breakdown chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Retrieval', x=df['Query'], y=df['Retrieval (ms)']))
        fig.add_trace(go.Bar(name='Generation', x=df['Query'], y=df['Generation (ms)']))
        fig.update_layout(
            barmode='stack',
            title='Latency Breakdown (Sample)',
            xaxis_title='Query',
            yaxis_title='Latency (ms)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Aggregate metrics from chat history
    history = st.session_state.chat_history
    
    # Summary metrics
    st.markdown("### 📊 Summary Statistics")
    st.caption("Aggregate metrics across all queries in this session")
    
    total_queries = len(history)
    avg_latency = sum(h['result']['metrics']['total_latency_ms'] for h in history) / total_queries
    avg_citations = sum(len(h['result']['citations']) for h in history) / total_queries
    avg_tokens = sum(h['result']['metrics']['total_tokens'] for h in history) / total_queries
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Queries", total_queries, help="Number of questions asked in this session")
    with col2:
        st.metric("⏱️ Avg Latency", f"{avg_latency:.0f}ms", help="Average total response time (retrieval + generation)")
    with col3:
        st.metric("📚 Avg Citations", f"{avg_citations:.1f}", help="Average number of review IDs cited per answer")
    with col4:
        st.metric("🔤 Avg Tokens", f"{avg_tokens:.0f}", help="Average tokens used per query (affects cost)")
    
    # Latency breakdown
    st.markdown("### ⏱️ Latency Analysis")
    st.caption("Breakdown of response time: retrieval (FAISS+BM25+MMR+Rerank) vs generation (GPT-4o-mini)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency breakdown stacked bar chart
        data = []
        for i, h in enumerate(history):
            data.append({
                'Query': f"Q{i+1}",
                'Retrieval': h['result']['metrics']['retrieval_latency_ms'],
                'Generation': h['result']['metrics']['generation_latency_ms']
            })
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Retrieval', x=df['Query'], y=df['Retrieval']))
        fig.add_trace(go.Bar(name='Generation', x=df['Query'], y=df['Generation']))
        fig.update_layout(
            barmode='stack',
            title='Latency Breakdown by Query',
            xaxis_title='Query',
            yaxis_title='Latency (ms)',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Total latency line chart
        latencies = [h['result']['metrics']['total_latency_ms'] for h in history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(latencies) + 1)),
            y=latencies,
            mode='lines+markers',
            name='Total Latency',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        fig.add_hline(y=avg_latency, line_dash="dash", line_color="red", annotation_text="Average")
        fig.update_layout(
            title='Total Latency Over Time',
            xaxis_title='Query Number',
            yaxis_title='Latency (ms)',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Citations and tokens
    st.markdown("### 📚 Citation & Token Analysis")
    st.caption("Citations indicate answer quality (more = better grounding). Tokens affect API cost (~$0.15/1M input, ~$0.60/1M output).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Citations per query
        citations = [len(h['result']['citations']) for h in history]
        
        fig = px.bar(
            x=list(range(1, len(citations) + 1)),
            y=citations,
            labels={'x': 'Query Number', 'y': 'Number of Citations'},
            title='Citations per Query'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Token usage
        token_data = []
        for i, h in enumerate(history):
            token_data.append({
                'Query': f"Q{i+1}",
                'Prompt': h['result']['metrics']['prompt_tokens'],
                'Completion': h['result']['metrics']['completion_tokens']
            })
        
        df = pd.DataFrame(token_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Prompt', x=df['Query'], y=df['Prompt']))
        fig.add_trace(go.Bar(name='Completion', x=df['Query'], y=df['Completion']))
        fig.update_layout(
            barmode='stack',
            title='Token Usage by Query',
            xaxis_title='Query',
            yaxis_title='Tokens',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Query details table
    st.markdown("### 📋 Query Details")
    st.caption("Complete history of all queries with performance metrics")
    
    table_data = []
    for i, h in enumerate(history):
        table_data.append({
            '#': i + 1,
            'Query': h['query'][:50] + "..." if len(h['query']) > 50 else h['query'],
            'Citations': len(h['result']['citations']),
            'Contexts': h['result']['metrics']['num_contexts'],
            'Total Time (ms)': f"{h['result']['metrics']['total_latency_ms']:.0f}",
            'Tokens': h['result']['metrics']['total_tokens']
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)


def main():
    """Main app entry point."""
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/3/3d/The_Walt_Disney_Company_logo.svg/1200px-The_Walt_Disney_Company_logo.svg.png", width=200)
        st.title("Disney RAG System")
        st.caption("v1.0.0")
        
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            **RAG Pipeline**:
            1. 🔍 Query Parser extracts filters
            2. 🔎 Hybrid Search (FAISS + BM25)
            3. 🎯 MMR Diversification
            4. 📊 Cross-Encoder Re-ranking
            5. 🤖 GPT-4o-mini Generation
            6. 📚 Citation Extraction
            
            **Data**: 42,656 reviews → 123,860 chunks
            
            **Performance**: ~8s latency, ~$0.0002/query
            """)
        
        with st.expander("🎯 How to Use"):
            st.markdown("""
            1. **Type a question** or click an example query
            2. **Adjust settings** if needed (number of chunks, filtering)
            3. **Click Search** and wait ~8 seconds
            4. **Read the answer** with citations
            5. **Check Eval tab** for analytics
            
            **Pro Tips**:
            - Mention parks: Hong Kong, Paris, California
            - Mention seasons: spring, summer, fall, winter
            - Mention countries: Australia, USA, UK, France
            - Ask about: staff, food, rides, crowds, value
            """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Evaluation", "🧪 Testing"])
    
    with tab1:
        chat_tab()
    
    with tab2:
        eval_tab()
    
    with tab3:
        testing_tab()
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit • RAG powered by FAISS + BM25 + GPT-4o-mini")


if __name__ == "__main__":
    main()

