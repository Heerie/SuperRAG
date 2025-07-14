# --- START OF FILE app.py ---

import streamlit as st
import os
import RAG2  # Import your RAG module
import logging
import torch

torch.__path__ = []
# --- Page Configuration ---
st.set_page_config(
    page_title="Document Query System",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Constants and Configuration ---
CONFIG_PATH = "config.ini"

# --- Logging ---
# The RAG2 module already configures logging. You can add Streamlit-specific logging if needed.
logger = logging.getLogger(__name__) # Get logger instance

# --- Environment Variable Check ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Initialization (Cached) ---
# Use st.cache_resource to initialize the RAG system only once and reuse it.
@st.cache_resource
def load_rag_system():
    """Loads or initializes the RAGSystem."""
    if not GROQ_API_KEY:
        st.error("❌ **Error:** GROQ_API_KEY environment variable is not set.")
        st.warning("Please set the GROQ_API_KEY environment variable and restart the application.")
        st.stop() # Halt execution if key is missing

    st.info("⏳ Initializing RAG System and processing PDFs... This might take a few moments.")
    try:
        rag_system = RAG2.RAGSystem(config_path=CONFIG_PATH)
        # Processing PDFs is crucial and should happen during initialization
        rag_system.process_pdfs()

        if not rag_system.faiss_indexes:
             st.warning("⚠️ No PDF documents were successfully processed or indexed.")
             st.info(f"Please check the '{rag_system.config.pdf_dir}' directory for PDFs and review 'rag_system.log' for errors.")
             # Return the system anyway, queries will likely fail gracefully or show no results
        else:
             st.success(f"✅ RAG System Ready. Indexed {len(rag_system.faiss_indexes)} document(s).")
        return rag_system
    except FileNotFoundError as e:
        st.error(f"❌ **Initialization Failed:** A required file or directory was not found: {e}")
        st.error(f"Please ensure configuration paths in '{CONFIG_PATH}' are correct.")
        logger.error(f"Initialization failed due to FileNotFoundError: {e}", exc_info=True)
        return None # Indicate failure
    except Exception as e:
        st.error(f"❌ **Critical Error during RAG System initialization:** {e}")
        st.error("Please check the 'rag_system.log' file for detailed error information.")
        logger.error(f"Unexpected error during RAG system initialization: {e}", exc_info=True)
        return None # Indicate failure


# --- Main App Logic ---
st.title("📄 Document Query System (RAG)")
st.markdown("Ask questions about your indexed PDF documents.")

# Load the RAG system using the cached function
rag_system_instance = load_rag_system()

# Only proceed if initialization was successful
if rag_system_instance:

    # Display configuration details (optional)
    with st.sidebar:
        st.header("Configuration")
        st.markdown(f"**LLM Model:** `{rag_system_instance.config.llm_model}`")
        st.markdown(f"**Encoder Model:** `{rag_system_instance.config.encoder_model}`")
        st.markdown(f"**PDF Directory:** `{rag_system_instance.config.pdf_dir}`")
        st.markdown(f"**Index Directory:** `{rag_system_instance.config.index_dir}`")
        st.markdown(f"**Retrieval K:** `{rag_system_instance.config.k_retrieval}`")
        st.markdown(f"**LLM Temperature:** `{rag_system_instance.config.temperature}`")
        st.caption("Settings loaded from config.ini")


    st.markdown("---")

    # --- Query Input ---
    st.header("❓ Ask your Question")
    user_query = st.text_area("Enter your query:", height=100, placeholder="e.g., What are the main safety procedures mentioned?")

    if st.button("🚀 Submit Query", type="primary"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a query.")
        elif not rag_system_instance.faiss_indexes:
             st.warning("⚠️ Cannot perform query: No documents were successfully indexed during initialization.")
        else:
            st.markdown("---")
            st.header("🔍 Results")
            with st.spinner("🧠 Thinking... Retrieving relevant information and generating answer..."):
                try:
                    # Run the query using the RAG system
                    results_data = rag_system_instance.run_query(user_query, context_strategy="top_k")

                    # Display Status
                    query_status = results_data.get("status", "Status unknown")
                    if "successfully" in query_status.lower():
                        st.success(f"✅ Status: {query_status}")
                    elif "no relevant chunks" in query_status.lower():
                         st.warning(f"⚠️ Status: {query_status}")
                    else:
                         st.info(f"ℹ️ Status: {query_status}")


                    # Display Generated Answers
                    answers = results_data.get("answers", {})
                    if answers:
                        st.subheader("💬 Generated Answer(s)")
                        for pdf_file, answer in answers.items():
                            st.markdown(f"**📄 Answer based on:** `{pdf_file}`")
                            # Use markdown blockquote for better visual separation
                            st.markdown(f"> {answer.strip()}")
                            st.markdown("---") # Separator between answers from different files
                    else:
                        st.warning("💬 No answer could be generated by the LLM based on the retrieved context.")

                    # Display Retrieval Details (in an expander)
                    retrieval_results = results_data.get("results", {})
                    if retrieval_results:
                        with st.expander("📚 Show Retrieved Context Chunks (Top Matches)", expanded=False):
                            st.markdown("_Lower score indicates a better semantic match (closer distance)._")
                            for pdf_file, pdf_res_list in retrieval_results.items():
                                if pdf_res_list:
                                    st.markdown(f"#### Relevant chunks from: `{pdf_file}`")
                                    # Display top N chunks (e.g., top 3)
                                    for i, res in enumerate(pdf_res_list[:3]):
                                        st.markdown(f"**Rank {i+1}:**")
                                        #st.markdown(f" - **Score:** `{res['score']:.4f}`")
                                        score_display = res.get('score', 'N/A') # Get score, default to 'N/A' if missing
                                        st.markdown(f" - **Score:** `{score_display if isinstance(score_display, (int, float)) else score_display}`")
                                        # Optional improvement: format only if it's a number
                                        if isinstance(score_display, (int, float)):
                                            st.markdown(f" - **Score:** `{score_display:.4f}`")
                                        else:
                                            st.markdown(f" - **Score:** `{score_display}`")
                                        st.markdown(f" - **Page:** `{res['page']}`")
                                        st.markdown(f" - **Type:** `{res['type']}`")
                                        # Use a smaller text display for the content snippet
                                        st.text_area(f"Content Snippet {i+1}:", value=res['content'].strip(), height=100, disabled=True, key=f"snippet_{pdf_file}_{i}")
                                        st.markdown("---") # Separator between chunks
                                # else: # Optionally indicate files with no results
                                #    st.markdown(f"_(No relevant chunks found in `{pdf_file}` for this query)_")
                    elif query_status != "No relevant chunks found.": # Avoid redundancy if no chunks were found at all
                         st.info("ℹ️ No specific context chunks were retrieved for display.")


                except Exception as e:
                    st.error(f"❌ **Error during query execution:** {e}")
                    logger.error(f"Error running Streamlit query '{user_query}': {e}", exc_info=True)
                    st.error("Check 'rag_system.log' for more details.")

elif not GROQ_API_KEY:
    # This message is shown if the API key check fails *before* initialization is attempted
    st.error("Application cannot start because the GROQ_API_KEY is missing.")
else:
    # This message covers other initialization failures after the API key check passed initially
    st.error("🔴 RAG System failed to initialize. Please check the logs and configuration, then restart the app.")


st.markdown("---")
st.caption("Check `rag_system.log` for detailed operational logs.")

# --- END OF FILE app.py ---