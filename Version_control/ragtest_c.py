# --- START OF FILE ragtest.py ---

import os
import re
import logging
import json
import configparser
import pdfplumber
import torch
import numpy as np
import pandas as pd
import faiss
import streamlit as st # Added for UI
import time # Added for potential delays/timing
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage # Keep for potential future use
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- START OF RAG SYSTEM CODE (Adapted from RAG2.py) ---

# Configure logging (optional for Streamlit, but good for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        # logging.StreamHandler() # Can be noisy in Streamlit console
    ]
)
logger = logging.getLogger(__name__)


class RAGConfig:
    """Configuration class for RAG system."""

    def __init__(self, config_path="config.ini"):
        """Initialize configuration from config file or defaults."""
        self.config = configparser.ConfigParser()

        # Default configuration
        self.defaults = {
            "PATHS": {
                "pdf_dir": "./Data/",
                "index_dir": "./Faiss_index/",
                "log_file": "rag_system.log"
            },
            "MODELS": {
                "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
                "summarizer_model": "t5-small",
                # Using a common Groq model name, adjust in config.ini if needed
                "llm_model": "llama3-8b-8192",
                "device": "auto"  # auto, cuda, or cpu (for encoder/summarizer)
            },
            "PARAMETERS": {
                "chunk_size": "200",
                "overlap": "50",
                "k_retrieval": "5",
                "temperature": "0.2" # Temperature for Groq
            }
        }

        # Try to load from config file, use defaults if not available
        if os.path.exists(config_path):
            try:
                self.config.read(config_path)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error reading config file {config_path}: {e}")
                self._set_defaults()
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self._set_defaults()
            # Don't automatically save in Streamlit env, might cause permission issues
            # self._save_config(config_path)

    def _set_defaults(self):
        """Set default configuration values."""
        for section, options in self.defaults.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for option, value in options.items():
                if not self.config.has_option(section, option):
                    self.config.set(section, option, value)

    # --- Properties (getters) for configuration values ---
    @property
    def pdf_dir(self): return self.config.get("PATHS", "pdf_dir")
    @property
    def index_dir(self): return self.config.get("PATHS", "index_dir")
    @property
    def log_file(self): return self.config.get("PATHS", "log_file")
    @property
    def encoder_model(self): return self.config.get("MODELS", "encoder_model")
    @property
    def summarizer_model(self): return self.config.get("MODELS", "summarizer_model")
    @property
    def llm_model(self): return self.config.get("MODELS", "llm_model")
    @property
    def device(self):
        device_setting = self.config.get("MODELS", "device")
        if device_setting == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting
    @property
    def chunk_size(self): return self.config.getint("PARAMETERS", "chunk_size")
    @property
    def overlap(self): return self.config.getint("PARAMETERS", "overlap")
    @property
    def k_retrieval(self): return self.config.getint("PARAMETERS", "k_retrieval")
    @property
    def temperature(self): return self.config.getfloat("PARAMETERS", "temperature")


class RAGSystem:
    """Retrieval-Augmented Generation system for PDF documents."""

    def __init__(self, config_path="config.ini"):
        """Initialize the RAG system with configuration."""
        self.config = RAGConfig(config_path)
        logger.info(f"Initializing RAG system with device for local models: {self.config.device}")

        # Create necessary directories
        os.makedirs(self.config.pdf_dir, exist_ok=True)
        os.makedirs(self.config.index_dir, exist_ok=True)

        # Initialize local models (encoder, summarizer) - Wrap in try-except for robustness
        try:
            # Use Streamlit spinner for model loading feedback during initialization
            with st.spinner(f"Loading embedding model ({self.config.encoder_model})..."):
                 self.encoder_model = SentenceTransformer(self.config.encoder_model).to(self.config.device)
            # Summarizer is not used in the final query flow, so loading can be deferred or skipped if memory is a concern
            # For now, we keep it as per the original code, but it could be removed if unused.
            # with st.spinner(f"Loading summarizer model ({self.config.summarizer_model})..."):
            #      self.summarizer_model = T5ForConditionalGeneration.from_pretrained(self.config.summarizer_model).to(self.config.device)
            #      self.tokenizer = T5Tokenizer.from_pretrained(self.config.summarizer_model)
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading local models: {e}", exc_info=True)
            st.error(f"Fatal Error: Failed to load local embedding models. Check logs. Error: {e}")
            # Stop streamlit execution if core models fail
            st.stop()
            # raise RuntimeError(f"Failed to initialize local models: {e}") # Alternative: Raise to stop completely


        # LLM (Groq) is initialized on-demand in query_llm using API key from .env

        # Initialize data containers
        self.pdf_chunks = {}
        self.faiss_indexes = {}
        self.processed_files = set() # Keep track of files processed in this session


    def clean_text(self, text):
        """Clean and normalize text."""
        if not text: return ""
        text = re.sub(r"\(cid:.*?\)", "", text) # Remove bad encoding chars
        text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
        text = text.replace('\n', ' ').replace('\r', '') # Remove newlines
        return text


    def extract_text_and_tables(self, pdf_path):
        """Extract text and tables from PDF files."""
        all_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Extracting from {pdf_path} ({total_pages} pages)")
                # Add progress bar for extraction if many pages
                page_bar = None
                if total_pages > 10: # Only show bar for larger docs
                    page_bar = st.progress(0, text=f"Extracting pages from {os.path.basename(pdf_path)}...")

                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text(layout="normal") or ""
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        all_content.append({"type": "text", "content": cleaned_text, "page": page_num + 1})

                    # Extract tables
                    try:
                        for table in page.extract_tables():
                             if table: # Ensure table is not empty
                                 table_df = pd.DataFrame(table).fillna('') # Handle None values
                                 # Simple string representation, consider markdown for better display later
                                 table_string = table_df.to_string(index=False, header=True)
                                 all_content.append({"type": "table", "content": table_string, "page": page_num + 1})
                    except Exception as e_table:
                         logger.warning(f"Could not extract table on page {page_num + 1} in {pdf_path}: {e_table}")

                    if page_bar:
                         progress_percent = min(1.0, (page_num + 1) / total_pages)
                         page_bar.progress(progress_percent, text=f"Extracting page {page_num + 1}/{total_pages} from {os.path.basename(pdf_path)}...")

                if page_bar: page_bar.empty() # Remove progress bar on completion

            logger.info(f"Extracted {len(all_content)} content blocks from {pdf_path}")
            return all_content
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {e}", exc_info=True)
            st.warning(f"Could not extract content from {os.path.basename(pdf_path)}. Error: {e}")
            return []


    def chunk_content(self, all_content):
        """Split content into chunks with overlap."""
        chunks = []
        words_processed = 0
        total_words_estimate = sum(len(item['content'].split()) for item in all_content) # Rough estimate

        chunk_bar = None
        if total_words_estimate > 5000: # Show progress for large content amounts
             chunk_bar = st.progress(0, text=f"Chunking content...")

        for item in all_content:
            content = item['content']
            page_num = item['page']
            content_type = item['type']
            words = content.split() # Simple whitespace split

            if not words: continue

            for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
                chunk_words = words[i:i + self.config.chunk_size]
                chunk = " ".join(chunk_words)
                if chunk:  # Ensure chunk is not empty
                    chunks.append({"content": chunk, "page": page_num, "type": content_type})

                # Update progress bar based on words processed in this item
                if chunk_bar:
                    words_processed += len(chunk_words)
                    progress_percent = min(1.0, words_processed / total_words_estimate) if total_words_estimate else 0
                    chunk_bar.progress(progress_percent, text=f"Chunking content... ({len(chunks)} chunks created)")

        if chunk_bar: chunk_bar.empty() # Remove progress bar

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    # --- Summarize Text (Not used in core query flow currently) ---
    # def summarize_text(self, text): ... (keep original code if needed, but likely remove T5 model loading if unused)

    # --- Path Helpers ---
    def _get_safe_filename(self, pdf_file):
        """Generate a safe filename for index/cache files."""
        # Replace .pdf and sanitize
        return re.sub(r'[^\w\.-]', '_', pdf_file.replace('.pdf', '').replace('.PDF', ''))

    def get_index_path(self, pdf_file):
        safe_name = self._get_safe_filename(pdf_file)
        return os.path.join(self.config.index_dir, f"{safe_name}.index")

    def get_embedding_path(self, pdf_file):
        safe_name = self._get_safe_filename(pdf_file)
        return os.path.join(self.config.index_dir, f"{safe_name}.npy")

    def get_chunks_path(self, pdf_file):
        safe_name = self._get_safe_filename(pdf_file)
        return os.path.join(self.config.index_dir, f"{safe_name}.json")

    # --- FAISS Index and Chunk Loading/Saving ---
    def load_faiss_index(self, pdf_file, embedding_dim):
        """Load FAISS index from disk or create new one."""
        index_path = self.get_index_path(pdf_file)
        if os.path.exists(index_path):
            try:
                logger.debug(f"Attempting to load FAISS index from {index_path}")
                index = faiss.read_index(index_path)
                if index.d != embedding_dim:
                    logger.warning(f"Index dimension mismatch for {pdf_file} (Expected {embedding_dim}, Got {index.d}). Rebuilding.")
                    return faiss.IndexFlatL2(embedding_dim)
                logger.info(f"Loaded FAISS index for {pdf_file} ({index.ntotal} vectors)")
                return index
            except Exception as e:
                logger.error(f"Error reading FAISS index {index_path}: {e}. Creating new index.")
                # Fall through to create a new index
        logger.info(f"FAISS index not found at {index_path}. Will create a new one.")
        return faiss.IndexFlatL2(embedding_dim)

    def save_chunks(self, pdf_file, chunks):
        """Save chunks to disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        try:
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2) # Smaller indent for space
            logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        except Exception as e:
            logger.error(f"Error saving chunks for {pdf_file} to {chunks_path}: {e}")
            st.warning(f"Could not save chunks for {os.path.basename(pdf_file)} to disk.")

    def load_chunks(self, pdf_file):
        """Load chunks from disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
                return chunks
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {chunks_path}: {e}. File might be corrupted. Will re-process.")
                return None # Force re-processing
            except Exception as e:
                logger.error(f"Error loading chunks from {chunks_path}: {e}")
                return None # Force re-processing
        logger.info(f"Chunks file not found at {chunks_path}")
        return None


    def process_pdfs(self, progress_callback=None):
        """Process all PDFs in directory, generating or loading embeddings and indexes."""
        # Reset in-memory stores
        self.pdf_chunks = {}
        self.faiss_indexes = {}
        self.processed_files = set()

        pdf_dir = self.config.pdf_dir
        try:
             all_files = os.listdir(pdf_dir)
             pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
             # Sort for consistent processing order
             pdf_files.sort()
        except FileNotFoundError:
             logger.error(f"PDF directory not found: {pdf_dir}")
             if progress_callback: progress_callback(f"❌ Error: PDF directory '{pdf_dir}' not found. Check config.ini.", is_error=True)
             return False # Indicate failure
        except Exception as e:
             logger.error(f"Error listing files in PDF directory {pdf_dir}: {e}")
             if progress_callback: progress_callback(f"❌ Error accessing PDF directory: {e}", is_error=True)
             return False

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            if progress_callback: progress_callback(f"⚠️ No PDF files found in '{pdf_dir}'. Add PDFs to process.", is_warning=True)
            # Return True because it's not an error, just nothing to do
            return True

        logger.info(f"Processing {len(pdf_files)} PDF files from {pdf_dir}")
        embedding_dim = self.encoder_model.get_sentence_embedding_dimension()

        total_files = len(pdf_files)
        if progress_callback: progress_callback(f"Found {total_files} PDF(s). Starting processing...", current_step=0, total_steps=total_files)

        # --- Main Processing Loop ---
        for idx, pdf_file in enumerate(pdf_files):
            current_file_msg = f"Processing ({idx+1}/{total_files}): {pdf_file}"
            if progress_callback: progress_callback(current_file_msg, current_step=idx, total_steps=total_files)

            logger.info(f"--- {current_file_msg} ---")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            index_path = self.get_index_path(pdf_file)
            emb_path = self.get_embedding_path(pdf_file)
            chunks_path = self.get_chunks_path(pdf_file)

            try:
                # 1. Load or Extract/Save Chunks
                chunks = self.load_chunks(pdf_file)
                if chunks is None: # If loading failed or file doesn't exist
                    if progress_callback: progress_callback(f"{current_file_msg} - Extracting content...", current_step=idx, total_steps=total_files, stage="Extracting")
                    logger.info(f"Extracting and chunking content for {pdf_file}...")
                    all_content = self.extract_text_and_tables(pdf_path)
                    if not all_content:
                         logger.warning(f"No content extracted from {pdf_file}. Skipping.")
                         if progress_callback: progress_callback(f"⚠️ No content extracted from {pdf_file}. Skipping.", is_warning=True, current_step=idx, total_steps=total_files)
                         continue # Skip to next file

                    if progress_callback: progress_callback(f"{current_file_msg} - Chunking content...", current_step=idx, total_steps=total_files, stage="Chunking")
                    chunks = self.chunk_content(all_content)
                    if not chunks:
                         logger.warning(f"No chunks generated for {pdf_file} after extraction. Skipping.")
                         if progress_callback: progress_callback(f"⚠️ No chunks generated for {pdf_file}. Skipping.", is_warning=True, current_step=idx, total_steps=total_files)
                         continue # Skip if chunking yields nothing
                    self.save_chunks(pdf_file, chunks) # Save newly created chunks

                # Store chunks in memory for this session
                self.pdf_chunks[pdf_file] = chunks
                logger.debug(f"Stored {len(chunks)} chunks in memory for {pdf_file}")


                # 2. Load or Generate/Save Embeddings and Index
                faiss_index = None
                regenerate_embeddings = False

                if os.path.exists(index_path) and os.path.exists(emb_path) and os.path.exists(chunks_path):
                    logger.info(f"Found existing index, embeddings, and chunks for {pdf_file}. Verifying...")
                    if progress_callback: progress_callback(f"{current_file_msg} - Verifying existing index...", current_step=idx, total_steps=total_files, stage="Verifying")
                    try:
                        embeddings = np.load(emb_path)
                        # Verify embedding dimensions
                        if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
                             logger.warning(f"Embeddings dimension mismatch for {pdf_file} (Shape: {embeddings.shape}, Expected Dim: {embedding_dim}). Regenerating.")
                             regenerate_embeddings = True
                        # Verify embedding count matches chunk count
                        elif embeddings.shape[0] != len(chunks):
                             logger.warning(f"Chunk/Embedding count mismatch for {pdf_file} ({len(chunks)} chunks vs {embeddings.shape[0]} embeddings). Regenerating.")
                             regenerate_embeddings = True
                        else:
                             # If dimensions and count match, try loading the index
                             faiss_index = self.load_faiss_index(pdf_file, embedding_dim)
                             # Verify the loaded index count matches
                             if faiss_index.ntotal != embeddings.shape[0]:
                                 logger.warning(f"FAISS index count ({faiss_index.ntotal}) doesn't match embeddings count ({embeddings.shape[0]}) for {pdf_file}. Rebuilding index from loaded embeddings.")
                                 # Rebuild index from existing embeddings
                                 faiss_index = faiss.IndexFlatL2(embedding_dim)
                                 faiss_index.add(embeddings.astype('float32'))
                                 faiss.write_index(faiss_index, index_path) # Save rebuilt index
                                 logger.info(f"Rebuilt and saved FAISS index for {pdf_file}")
                             else:
                                 logger.info(f"Successfully loaded and verified existing embeddings ({embeddings.shape[0]} vectors) and index for {pdf_file}")

                    except Exception as e:
                        logger.error(f"Error loading/verifying pre-existing index/embeddings for {pdf_file}: {e}. Regenerating...", exc_info=True)
                        regenerate_embeddings = True # Force regeneration on any error
                        faiss_index = None # Ensure regeneration logic runs

                else: # If any file (index, embeddings, chunks) is missing
                     logger.info(f"Required files missing for {pdf_file}. Generating embeddings and index.")
                     regenerate_embeddings = True


                # If index/embeddings need regeneration
                if regenerate_embeddings or faiss_index is None:
                    logger.info(f"Generating embeddings and creating/updating index for {pdf_file}...")
                    if progress_callback: progress_callback(f"{current_file_msg} - Generating embeddings...", current_step=idx, total_steps=total_files, stage="Embedding")

                    # Prepare texts for batch encoding
                    content_list = [chunk['content'] for chunk in chunks]
                    if not content_list:
                         logger.warning(f"No content to embed for {pdf_file}. Skipping embedding/indexing.")
                         continue

                    # Batch encode for efficiency
                    embeddings = self.encoder_model.encode(
                        content_list,
                        batch_size=64, # Increase batch size if GPU allows
                        show_progress_bar=False, # Disable internal bar, use Streamlit feedback
                        convert_to_numpy=True
                    ).astype('float32')

                    if embeddings.shape[0] == 0:
                        logger.warning(f"Embedding process yielded no vectors for {pdf_file}. Skipping index creation.")
                        continue

                    # Save embeddings
                    np.save(emb_path, embeddings)
                    logger.info(f"Saved {embeddings.shape[0]} embeddings to {emb_path}")

                    # Create and save FAISS index
                    if progress_callback: progress_callback(f"{current_file_msg} - Creating FAISS index...", current_step=idx, total_steps=total_files, stage="Indexing")
                    faiss_index = faiss.IndexFlatL2(embedding_dim)
                    faiss_index.add(embeddings)
                    faiss.write_index(faiss_index, index_path)
                    logger.info(f"Created and saved FAISS index with {faiss_index.ntotal} vectors to {index_path}")

                # Store the final loaded/created index in memory
                if faiss_index is not None and faiss_index.ntotal > 0:
                    self.faiss_indexes[pdf_file] = faiss_index
                    self.processed_files.add(pdf_file) # Mark as successfully processed
                    logger.debug(f"Stored FAISS index in memory for {pdf_file}")
                else:
                    logger.warning(f"No valid FAISS index generated or loaded for {pdf_file}. It will not be searchable.")
                    if progress_callback: progress_callback(f"⚠️ Could not create/load index for {pdf_file}.", is_warning=True, current_step=idx, total_steps=total_files)


            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}", exc_info=True)
                if progress_callback: progress_callback(f"❌ Error processing {pdf_file}: {e}", is_error=True, current_step=idx, total_steps=total_files)
                # Attempt cleanup of potentially corrupted intermediate files for *this* PDF
                try:
                    if os.path.exists(index_path): os.remove(index_path)
                    if os.path.exists(emb_path): os.remove(emb_path)
                    # Keep chunks file? Might be reusable if only embedding failed. Optional: remove chunks_path too.
                    # if os.path.exists(chunks_path): os.remove(chunks_path)
                    logger.info(f"Cleaned up index/embedding files for failed {pdf_file}")
                except OSError as e_clean:
                    logger.error(f"Error cleaning up files for {pdf_file}: {e_clean}")
                # Remove from in-memory stores if partially added
                if pdf_file in self.pdf_chunks: del self.pdf_chunks[pdf_file]
                if pdf_file in self.faiss_indexes: del self.faiss_indexes[pdf_file]


        # --- Post-processing Summary ---
        final_indexed_count = len(self.faiss_indexes)
        if final_indexed_count > 0:
             logger.info(f"--- PDF Processing Complete. Indexed {final_indexed_count}/{total_files} documents. ---")
             if progress_callback: progress_callback(f"✅ Processing Complete. Ready to query {final_indexed_count} document(s).", current_step=total_files, total_steps=total_files, is_done=True)
             return True # Success
        else:
             logger.warning(f"--- PDF Processing Complete. No documents were successfully indexed. ---")
             if progress_callback: progress_callback(f"⚠️ Processing Finished. No documents could be indexed. Check logs and PDF files.", is_warning=True, is_done=True)
             return False # Failure


    def query_pdfs(self, query):
        """Query all processed PDFs with a user query."""
        if not self.faiss_indexes:
            logger.warning("No indexes available. Run process_pdfs() first or check processing errors.")
            st.warning("No documents seem to be indexed. Please process PDFs first or check logs.")
            return {}

        all_results = {}
        try:
            logger.info(f"Encoding query: '{query[:100]}...'")
            query_embedding = self.encoder_model.encode(query, convert_to_numpy=True).astype("float32")
            # Reshape for FAISS search (expects batch dimension: [1, embedding_dim])
            query_embedding = np.array([query_embedding])
            if query_embedding.ndim != 2:
                 logger.error(f"Query embedding has unexpected shape: {query_embedding.shape}")
                 st.error("Internal error: Could not prepare query embedding.")
                 return {}


            logger.info(f"Searching across {len(self.faiss_indexes)} indexed documents.")
            # Add search progress if many indexes
            search_bar = None
            if len(self.faiss_indexes) > 5:
                 search_bar = st.progress(0, text=f"Searching documents...")

            processed_index_count = 0
            total_indexes = len(self.faiss_indexes)

            for pdf_file, index in self.faiss_indexes.items():
                pdf_results = []
                # Ensure index exists and is not empty
                if index is None or index.ntotal == 0:
                    logger.warning(f"Index for {pdf_file} is empty or invalid. Skipping search.")
                    continue

                # Update progress
                processed_index_count += 1
                if search_bar:
                     progress_percent = min(1.0, processed_index_count / total_indexes)
                     search_bar.progress(progress_percent, text=f"Searching {pdf_file} ({processed_index_count}/{total_indexes})...")


                try:
                    # Determine k: min of configured k, and actual number of items in index
                    k_search = min(self.config.k_retrieval, index.ntotal)
                    logger.debug(f"Searching index for {pdf_file} (k={k_search}, index size={index.ntotal})")

                    # D: distances (L2 squared), I: indices
                    D, I = index.search(query_embedding, k=k_search)

                    # Process search results (I[0] contains indices for the first query vector)
                    indices = I[0]
                    distances = D[0]
                    logger.debug(f"Retrieved {len(indices)} results for {pdf_file}. Indices: {indices}, Distances: {distances}")

                    # Retrieve corresponding chunks
                    current_pdf_chunks = self.pdf_chunks.get(pdf_file)
                    if not current_pdf_chunks:
                         logger.error(f"FATAL: Chunks for {pdf_file} not found in memory despite having an index! Cannot retrieve content.")
                         # This shouldn't happen if process_pdfs worked correctly
                         st.error(f"Internal inconsistency: Chunk data missing for {pdf_file}. Skipping.")
                         continue

                    processed_indices = set() # Avoid duplicates if k > ntotal somehow
                    for i, idx in enumerate(indices):
                        # FAISS might return -1 if k > ntotal or other issues
                        if idx == -1 or idx in processed_indices: continue
                        # Bounds check for safety
                        if not (0 <= idx < len(current_pdf_chunks)):
                             logger.warning(f"Retrieved invalid index {idx} for {pdf_file} (max: {len(current_pdf_chunks)-1}). Skipping.")
                             continue

                        processed_indices.add(idx)
                        chunk = current_pdf_chunks[idx]
                        distance = distances[i]
                        # Score is raw L2 distance (lower is better)
                        score = float(distance)

                        pdf_results.append({
                            "page": chunk.get('page', 'N/A'),
                            "content": chunk.get('content', ''),
                            "score": round(score, 4), # Lower score = better match
                            "type": chunk.get('type', 'unknown')
                        })

                    # Sort results by score (ascending L2 distance)
                    pdf_results.sort(key=lambda x: x['score'])
                    if pdf_results: # Only add if we found something
                        all_results[pdf_file] = pdf_results
                        logger.debug(f"Found {len(pdf_results)} relevant chunks in {pdf_file}")
                    else:
                         logger.debug(f"No relevant chunks found meeting criteria in {pdf_file}")


                except Exception as e_search:
                    logger.error(f"Error searching index for {pdf_file}: {e_search}", exc_info=True)
                    st.warning(f"Could not search {os.path.basename(pdf_file)}. Error: {e_search}")

            if search_bar: search_bar.empty() # Remove search progress bar

            logger.info(f"Search complete. Found relevant results in {len(all_results)} documents.")
            return all_results

        except Exception as e_query:
            logger.error(f"Error during query encoding or overall search processing: {e_query}", exc_info=True)
            st.error(f"An error occurred during the search: {e_query}")
            return {}


    def aggregate_context(self, query_results, strategy="top_k", max_context_tokens=4000):
        """Aggregate context from query results for LLM, respecting token limits."""
        all_context = {}
        # Estimate max chars: 1 token ~ 3-4 chars. Use 3 for safety margin.
        max_chars = max_context_tokens * 3
        logger.info(f"Aggregating context using strategy '{strategy}', max_chars ~{max_chars}")

        if not query_results:
            logger.warning("No query results provided to aggregate context.")
            return all_context

        total_aggregated_chars = 0

        for pdf_file, results in query_results.items():
            if not results:
                logger.debug(f"No results for {pdf_file} to aggregate.")
                continue

            # Results are pre-sorted by score (lower is better) in query_pdfs
            context = ""
            current_chars = 0
            num_chunks_added = 0

            # Determine how many results to consider based on strategy
            limit = self.config.k_retrieval if strategy == "top_k" else len(results)

            for i, res in enumerate(results[:limit]):
                # Format content with metadata
                content_header = f"[Source: {pdf_file}, Page: {res['page']}, Type: {res['type']}, Score: {res['score']:.4f}]\n"
                content_body = res['content']
                content_to_add = content_header + content_body + "\n\n"
                content_chars = len(content_to_add)

                # Check if adding this chunk exceeds the limit
                if current_chars + content_chars <= max_chars:
                    context += content_to_add
                    current_chars += content_chars
                    num_chunks_added += 1
                else:
                    # If even the first chunk is too large, truncate it severely
                    if i == 0:
                         remaining_chars = max_chars - current_chars - len(content_header) - len("\n\n[...TRUNCATED...]\n\n")
                         if remaining_chars > 50: # Only add if meaningful truncation possible
                             truncated_body = content_body[:remaining_chars]
                             context += content_header + truncated_body + "\n[...TRUNCATED...]\n\n"
                             current_chars += len(content_header) + len(truncated_body) + len("\n[...TRUNCATED...]\n\n")
                             num_chunks_added += 1
                             logger.warning(f"Context for {pdf_file} truncated (first chunk too large).")
                         else:
                              logger.warning(f"First chunk for {pdf_file} too large to fit meaningfully, skipping context.")
                    # Stop adding more chunks if limit reached
                    logger.info(f"Stopping context aggregation for {pdf_file} at {num_chunks_added} chunks due to character limit ({current_chars}/{max_chars} chars).")
                    break

            final_context = context.strip()
            if final_context:
                 all_context[pdf_file] = final_context
                 total_aggregated_chars += len(final_context)
                 logger.debug(f"Aggregated {len(final_context)} chars of context for {pdf_file} from {num_chunks_added} chunks.")
            else:
                 logger.warning(f"No context could be aggregated for {pdf_file} within limits.")

        logger.info(f"Total aggregated context characters across all files: {total_aggregated_chars}")
        return all_context


    def query_llm(self, query, context, pdf_file_source, retry_count=1):
        """Query the Groq API with context and handle errors."""
        if not context:
            logger.warning(f"No context provided for LLM query regarding {pdf_file_source}")
            return f"Could not generate answer for {pdf_file_source}: No relevant context found or aggregated."

        # Get API Key from environment - check should happen earlier, but double-check
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable not found.")
            st.error("Error: GROQ_API_KEY is not set in the environment.")
            return "Error: Groq API key not configured."

        try:
            # Initialize Groq Chat Model - consider caching this per request if making many calls
            llm = ChatGroq(
                temperature=self.config.temperature,
                model_name=self.config.llm_model,
                groq_api_key=api_key,
                # Add timeout?
                # request_timeout=30, # Example: 30 seconds timeout
            )
            
            # Prepare the prompt
            # System prompt emphasizes sticking to context
            system_prompt = f"""
            You are called the SatSure LLM. You answer questions that customers ask about this company. You are given the proposals and request for proposals of multiple projects combined into one file ({pdf_file_source}). Use this to infer and understand the entire document and then answer the question accordingly.  
            You are an expert assistant specialized in answering questions based *only* on the provided context from a specific PDF document ({pdf_file_source}).
Your task is to synthesize information found *solely* within the given text excerpts.
- Answer the user's question accurately using *only* the information present in the context below.
- If the context does not contain the information needed to answer the question, you MUST state: "Based on the provided context from '{pdf_file_source}', I cannot answer this question."
- Do NOT use any external knowledge, make assumptions, or infer information not explicitly stated in the context.
- Be concise and directly answer the question.
- Quote relevant parts of the context sparingly if it helps clarify the answer, but prioritize synthesizing the information.
- Reference the source document name ('{pdf_file_source}') in your response if you cannot answer.
"""
            human_prompt = f"""Context from document '{pdf_file_source}':
--- START CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {query}

Answer based *only* on the provided context:"""

            full_prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]


            logger.info(f"Querying Groq model {self.config.llm_model} for {pdf_file_source}...")
            # logger.debug(f"Prompt for {pdf_file_source}:\nSystem: {system_prompt}\nUser: {human_prompt[:500]}...") # Log truncated prompt

            answer = "Error: LLM query failed after retries." # Default error message
            for attempt in range(retry_count + 1):
                try:
                    response = llm.invoke(full_prompt_messages) # Use invoke with messages
                    # Check response type (might be AIMessage or string depending on Langchain version)
                    if hasattr(response, 'content'):
                         answer = response.content.strip()
                    elif isinstance(response, str):
                         answer = response.strip()
                    else:
                         logger.error(f"Unexpected response type from Groq: {type(response)}")
                         answer = "Error: Received unexpected response format from LLM."

                    logger.info(f"Groq response received for {pdf_file_source} (attempt {attempt+1}). Length: {len(answer)}")
                    # Basic check for refusal
                    if "cannot answer this question" in answer.lower() and "based on the provided context" in answer.lower():
                        logger.info(f"LLM indicated it could not answer based on context for {pdf_file_source}.")
                    # Add more checks? e.g., empty response
                    if not answer:
                         logger.warning(f"Received empty response from Groq for {pdf_file_source} on attempt {attempt+1}")
                         answer = f"Received an empty response from the language model for {pdf_file_source}."
                         # Optionally retry on empty response?
                    return answer # Success

                except Exception as e_api:
                    logger.warning(f"Groq API query attempt {attempt+1} for {pdf_file_source} failed: {e_api}")
                    if attempt < retry_count:
                        logger.info(f"Retrying Groq query for {pdf_file_source}...")
                        time.sleep(1.5 ** attempt) # Exponential backoff
                        continue # Retry the loop
                    else:
                        logger.error(f"All Groq API query attempts failed for {pdf_file_source}: {e_api}", exc_info=True)
                        answer = f"Error: Failed to get answer from language model for {pdf_file_source} after {retry_count+1} attempts. (Check Logs for details: {e_api})"
                        # Return the error instead of fallback
                        return answer


            # This part should ideally not be reached if retry logic is correct
            return answer

        except Exception as e_setup:
            logger.error(f"Error setting up or calling Groq API for {pdf_file_source}: {e_setup}", exc_info=True)
            st.error(f"Failed to query the language model for {pdf_file_source}. Error: {e_setup}")
            return f"Error: Could not query language model for {pdf_file_source} due to setup error: {e_setup}"

    # Fallback response generation is integrated into query_llm failure modes now.
    # def get_fallback_response(self, query, pdf_file_source, error=None): ...


    def run_query(self, query, context_strategy="top_k"):
        """Complete query pipeline: retrieve, aggregate context, query LLM."""
        logger.info(f"--- Starting query execution for: '{query[:100]}...' ---")
        final_results = {
            "query": query,
            "retrieval_results": {},
            "aggregated_context": {},
            "answers": {},
            "status": "Started",
            "error": None
        }

        try:
            # 1. Retrieve relevant chunks
            with st.spinner("Searching relevant documents..."):
                 query_results = self.query_pdfs(query)
                 final_results["retrieval_results"] = query_results

            if not query_results:
                logger.warning("No relevant chunks found for the query across any documents.")
                final_results["status"] = "Completed: No relevant information found."
                st.info("Could not find relevant information in the indexed documents for your query.")
                return final_results

            # 2. Aggregate context
            with st.spinner("Gathering context for answering..."):
                contexts = self.aggregate_context(query_results, strategy=context_strategy)
                final_results["aggregated_context"] = contexts # Store context for debugging/inspection

            if not contexts:
                 logger.warning(f"Context aggregation failed or produced no content for query: '{query[:100]}...'")
                 final_results["status"] = "Completed: Found relevant parts, but could not prepare context for answering."
                 final_results["error"] = "Context aggregation failed."
                 st.warning("Found some relevant document parts, but couldn't prepare them for the AI to answer. This might happen if relevant snippets are too long.")
                 return final_results

            # 3. Query LLM (Groq) for each document's context
            answers = {}
            num_contexts = len(contexts)
            llm_bar = st.progress(0, text=f"Generating answers (0/{num_contexts})...")
            processed_count = 0

            for pdf_file, context in contexts.items():
                processed_count += 1
                llm_bar.progress(min(1.0, processed_count / num_contexts), text=f"Generating answer based on {pdf_file} ({processed_count}/{num_contexts})...")
                logger.info(f"Generating answer for {pdf_file}...")
                # Pass pdf_file_source to query_llm
                answer = self.query_llm(query, context, pdf_file_source=pdf_file)
                answers[pdf_file] = answer
                logger.info(f"Answer received for {pdf_file}.")

            final_results["answers"] = answers
            llm_bar.empty() # Remove progress bar

            final_results["status"] = "Completed Successfully."
            logger.info(f"--- Finished query execution for: '{query[:100]}...' ---")

        except Exception as e:
            logger.error(f"An unexpected error occurred during run_query: {e}", exc_info=True)
            final_results["status"] = "Failed"
            final_results["error"] = str(e)
            st.error(f"An unexpected error occurred: {e}")

        return final_results

# --- END OF RAG SYSTEM CODE ---


# --- START OF STREAMLIT UI CODE ---

st.set_page_config(layout="wide", page_title="Document Q&A with RAG")

st.title("📄 Document Question Answering System")
st.caption("Powered by Local Embeddings, FAISS, and Groq LLM")

# --- Initialization and Caching ---
@st.cache_resource # Cache the RAG system instance
def load_rag_system(config_path="config.ini"):
    """Loads the RAGSystem, processing PDFs only on first load or if cache is cleared."""
    # Check for API key before initializing
    if not os.getenv("GROQ_API_KEY"):
        st.error("🚨 GROQ_API_KEY environment variable not set! Please set it in your .env file or environment.")
        st.stop() # Stop execution if key is missing

    rag_system = RAGSystem(config_path=config_path)
    return rag_system

@st.cache_data # Cache the result of PDF processing status
def process_documents(_rag_system):
    """Processes PDFs and returns status. Caching prevents reprocessing unless PDFs change."""
    status_messages = []
    processing_successful = False

    def streamlit_progress_callback(message, is_error=False, is_warning=False, is_done=False, current_step=None, total_steps=None, stage=None):
        """Callback function to update Streamlit UI during processing."""
        log_prefix = ""
        if is_error: log_prefix = "❌ Error: "
        elif is_warning: log_prefix = "⚠️ Warning: "
        elif is_done: log_prefix = "✅ "

        full_message = f"{log_prefix}{message}"
        status_messages.append(full_message) # Keep history

        # Update placeholder with the latest message and progress
        if total_steps and current_step is not None:
             progress_percent = min(1.0, (current_step + (0.5 if stage else 1) ) / total_steps) if total_steps > 0 else 0
             status_placeholder.progress(progress_percent)
             status_placeholder.caption(full_message)
        else:
             status_placeholder.caption(full_message) # Show message without progress bar

        # Log to console/file as well
        if is_error: logger.error(message)
        elif is_warning: logger.warning(message)
        else: logger.info(message)


    st.header("📚 Document Processing")
    status_placeholder = st.empty() # Create placeholder for status updates/progress bar

    with st.spinner("Initializing and processing documents... This may take a while on first run."):
        try:
            processing_successful = _rag_system.process_pdfs(progress_callback=streamlit_progress_callback)
            # Display final summary message
            final_message = status_messages[-1] if status_messages else "Processing state unknown."
            if processing_successful and _rag_system.faiss_indexes:
                status_placeholder.success(f"✅ Ready! Processed and indexed {len(_rag_system.faiss_indexes)} document(s).")
            elif _rag_system.processed_files and not _rag_system.faiss_indexes:
                 status_placeholder.warning(f"⚠️ Processing finished, but no documents were successfully indexed. Check logs.")
            elif not _rag_system.processed_files:
                 status_placeholder.warning(f"⚠️ No PDF documents found or processed in '{_rag_system.config.pdf_dir}'.")

        except Exception as e:
            logger.error(f"Fatal error during RAG system initialization or PDF processing: {e}", exc_info=True)
            status_placeholder.error(f"❌ A fatal error occurred during setup: {e}. Please check the logs.")
            processing_successful = False

    # Return status and the list of indexed files for display
    return processing_successful, list(_rag_system.faiss_indexes.keys())


# --- Main App Logic ---
try:
    rag_sys = load_rag_system() # Load cached RAG system

    # Process documents and get status (uses caching)
    is_ready, indexed_files = process_documents(rag_sys)

    if is_ready and indexed_files:
        st.sidebar.success(f"Indexed Documents ({len(indexed_files)}):")
        # Use an expander in the sidebar to list files without cluttering
        with st.sidebar.expander("Show Indexed Files"):
             for fname in indexed_files:
                 st.caption(f"- {fname}")
        st.sidebar.info(f"Using LLM: `{rag_sys.config.llm_model}`")
        st.sidebar.info(f"Retrieval K: `{rag_sys.config.k_retrieval}`")

        st.header("💬 Ask a Question")
        user_query = st.text_input("Enter your query about the indexed documents:", key="query_input")

        if user_query:
            if st.button("Get Answer", key="submit_query"):
                with st.spinner("Thinking... (Retrieving relevant info & Querying LLM)"):
                    start_time = time.time()
                    results_data = rag_sys.run_query(user_query, context_strategy="top_k")
                    end_time = time.time()

                st.subheader("💡 Answer(s)")
                if results_data and results_data.get("answers"):
                    for pdf_file, answer in results_data["answers"].items():
                        with st.expander(f"Answer based on: **{pdf_file}**", expanded=True):
                            st.markdown(answer) # Display LLM answer
                else:
                    st.warning("Could not generate an answer. This might be because no relevant information was found, the context was insufficient, or an error occurred.")
                    if results_data.get("error"):
                        st.error(f"Error details: {results_data['error']}")


                # Optional: Display Retrieval Details
                st.subheader("🔍 Retrieval Details (Supporting Evidence)")
                retrieval_results = results_data.get("retrieval_results", {})
                if retrieval_results:
                     for pdf_file, pdf_res_list in retrieval_results.items():
                         if pdf_res_list:
                             with st.expander(f"Top {len(pdf_res_list)} relevant chunks from: **{pdf_file}**"):
                                 for i, res in enumerate(pdf_res_list):
                                     st.markdown(f"**Chunk {i+1} (Score: {res['score']:.4f} - Lower is better)**")
                                     st.caption(f"Page: {res['page']} | Type: {res['type']}")
                                     # Display content snippet - use st.text or st.markdown
                                     st.text(f"{res['content'][:300]}...") # Show preview
                                     st.divider()
                         # else: # Optionally mention files with no hits
                         #    st.caption(f"No relevant chunks found in {pdf_file} for this query.")

                else:
                     st.info("No specific document chunks were retrieved for this query.")

                st.caption(f"Query processed in {end_time - start_time:.2f} seconds.")


    elif not indexed_files:
        st.warning("No documents are currently indexed. Please add PDF files to the specified data directory and refresh.")
        st.info(f"Looking for PDFs in: `{os.path.abspath(rag_sys.config.pdf_dir)}`")
        st.info(f"Storing index files in: `{os.path.abspath(rag_sys.config.index_dir)}`")
    else: # Not ready, and processing failed
         st.error("The RAG system could not be initialized or documents could not be processed. Please check the logs (`rag_system.log`) for more details.")


except Exception as e:
    st.error(f"An unexpected error occurred in the Streamlit application: {e}")
    logger.error(f"Streamlit application error: {e}", exc_info=True)
    st.info("Please check the console or logs for more details.")


# --- END OF STREAMLIT UI CODE ---

# --- END OF FILE ragtest.py ---