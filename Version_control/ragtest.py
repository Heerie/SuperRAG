# --- START OF FILE RAG2.py ---

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
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Removed Ollama import
# from langchain_community.llms import Ollama
# Added Groq import
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage # To handle response object if needed
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
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
                 # Changed default to a Groq model, update in config.ini as needed
                "llm_model": "meta-llama/llama-4-scout-17b-16e-instruct",
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
                logger.error(f"Error reading config file: {e}")
                self._set_defaults()
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self._set_defaults()
            self._save_config(config_path)

    def _set_defaults(self):
        """Set default configuration values."""
        for section, options in self.defaults.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for option, value in options.items():
                if not self.config.has_option(section, option):
                    self.config.set(section, option, value)

    def _save_config(self, config_path):
        """Save current configuration to file."""
        try:
            with open(config_path, 'w') as f:
                self.config.write(f)
            logger.info(f"Saved default configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")

    @property
    def pdf_dir(self):
        return self.config.get("PATHS", "pdf_dir")

    @property
    def index_dir(self):
        return self.config.get("PATHS", "index_dir")

    @property
    def log_file(self):
        return self.config.get("PATHS", "log_file")

    @property
    def encoder_model(self):
        return self.config.get("MODELS", "encoder_model")

    @property
    def summarizer_model(self):
        return self.config.get("MODELS", "summarizer_model")

    @property
    def llm_model(self):
        # This now refers to the Groq model name
        return self.config.get("MODELS", "llm_model")

    @property
    def device(self):
        # Device setting primarily for local models (encoder, summarizer)
        device_setting = self.config.get("MODELS", "device")
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    @property
    def chunk_size(self):
        return self.config.getint("PARAMETERS", "chunk_size")

    @property
    def overlap(self):
        return self.config.getint("PARAMETERS", "overlap")

    @property
    def k_retrieval(self):
        return self.config.getint("PARAMETERS", "k_retrieval")

    @property
    def temperature(self):
        # Temperature for the LLM (Groq)
        return self.config.getfloat("PARAMETERS", "temperature")


class RAGSystem:
    """Retrieval-Augmented Generation system for PDF documents."""

    def __init__(self, config_path="config.ini"):
        """Initialize the RAG system with configuration."""
        self.config = RAGConfig(config_path)
        logger.info(f"Initializing RAG system with device for local models: {self.config.device}")

        # Create necessary directories
        os.makedirs(self.config.pdf_dir, exist_ok=True)
        os.makedirs(self.config.index_dir, exist_ok=True)

        # Initialize local models (encoder, summarizer)
        try:
            self.encoder_model = SentenceTransformer(self.config.encoder_model).to(self.config.device)
            self.summarizer_model = T5ForConditionalGeneration.from_pretrained(self.config.summarizer_model).to(self.config.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.summarizer_model)
            logger.info("Local models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading local models: {e}")
            raise RuntimeError(f"Failed to initialize local models: {e}")

        # LLM (Groq) is initialized on-demand in query_llm using API key

        # Initialize data containers
        self.pdf_chunks = {}
        self.faiss_indexes = {}

    def clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r"\(cid:.*?\)", "", text)  # Remove bad encoding
        text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
        text = text.replace('\n', ' ').replace('\r', '')  # Remove newline and carriage return
        return text

    def extract_text_and_tables(self, pdf_path):
        """Extract text and tables from PDF files."""
        all_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text(layout="normal") or ""
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        all_content.append({
                            "type": "text",
                            "content": cleaned_text,
                            "page": page_num + 1
                        })

                    # Extract tables
                    for table in page.extract_tables():
                        try:
                            table_df = pd.DataFrame(table)
                            # Replace None with empty string for cleaner to_string() output
                            table_df = table_df.fillna('')
                            table_string = table_df.to_string(index=False, header=True) # More readable table string
                            all_content.append({
                                "type": "table",
                                "content": table_string,
                                "page": page_num + 1
                            })
                        except Exception as e:
                            logger.warning(f"Error processing table on page {page_num + 1} in {pdf_path}: {e}")

            logger.info(f"Extracted {len(all_content)} content blocks from {pdf_path}")
            return all_content
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {e}")
            return []

    def chunk_content(self, all_content):
        """Split content into chunks with overlap."""
        chunks = []
        for item in all_content:
            content = item['content']
            page_num = item['page']
            content_type = item['type']
            # Use split preserving spaces for better reconstruction, handle tables as single units if small enough
            # For simplicity here, we stick to word splitting
            words = content.split()

            if not words:
                continue

            for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
                chunk_words = words[i:i + self.config.chunk_size]
                chunk = " ".join(chunk_words)
                if chunk:  # Only add non-empty chunks
                    chunks.append({
                        "content": chunk,
                        "page": page_num,
                        "type": content_type
                    })

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def summarize_text(self, text):
        """Summarize text using T5 model."""
        if not text or len(text.strip()) < 10:
            return "No content to summarize"

        try:
            # Ensure input text fits within model's max length
            input_text = "summarize: " + text[:1024] # Limit input length for T5-small robustness
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.config.device)
            # Adjust generation parameters if needed
            summary_ids = self.summarizer_model.generate(
                input_ids,
                max_length=150, # Max length of summary
                min_length=30,  # Min length of summary
                num_beams=4,
                early_stopping=True,
                length_penalty=2.0 # Penalize longer summaries slightly
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip() if summary else "Summary could not be generated."
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            # Provide a more specific fallback
            return f"Failed to generate summary due to an error: {e}"


    def get_index_path(self, pdf_file):
        """Get path for FAISS index file."""
        # Sanitize filename for path creation
        safe_filename = re.sub(r'[^\w\.-]', '_', pdf_file.replace('.pdf', ''))
        return os.path.join(self.config.index_dir, f"{safe_filename}.index")

    def get_embedding_path(self, pdf_file):
        """Get path for embedding numpy file."""
        safe_filename = re.sub(r'[^\w\.-]', '_', pdf_file.replace('.pdf', ''))
        return os.path.join(self.config.index_dir, f"{safe_filename}.npy")

    def get_chunks_path(self, pdf_file):
        """Get path for chunks JSON file."""
        safe_filename = re.sub(r'[^\w\.-]', '_', pdf_file.replace('.pdf', ''))
        return os.path.join(self.config.index_dir, f"{safe_filename}.json")


    def load_faiss_index(self, pdf_file, embedding_dim):
        """Load FAISS index from disk or create new one."""
        index_path = self.get_index_path(pdf_file)
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                # Verify dimension compatibility if possible (though IndexFlatL2 might not store it directly)
                if index.d != embedding_dim:
                     logger.warning(f"Index dimension mismatch for {pdf_file} (Expected {embedding_dim}, Got {index.d}). Rebuilding.")
                     return faiss.IndexFlatL2(embedding_dim) # Return new index if mismatch
                logger.info(f"Loaded FAISS index from {index_path}")
                return index
            except Exception as e:
                logger.error(f"Error reading FAISS index {index_path}: {e}. Creating new index.")
                # Fall through to create a new index

        # Create new index if not found or error loading
        logger.info(f"Creating new FAISS index for {pdf_file} with dimension {embedding_dim}")
        return faiss.IndexFlatL2(embedding_dim)

    def save_chunks(self, pdf_file, chunks):
        """Save chunks to disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        try:
            with open(chunks_path, 'w', encoding='utf-8') as f: # Specify encoding
                json.dump(chunks, f, indent=4) # Use indent for readability
            logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        except Exception as e:
            logger.error(f"Error saving chunks for {pdf_file} to {chunks_path}: {e}")

    def load_chunks(self, pdf_file):
        """Load chunks from disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f: # Specify encoding
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
                return chunks
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {chunks_path}: {e}. File might be corrupted.")
                return None
            except Exception as e:
                logger.error(f"Error loading chunks from {chunks_path}: {e}")
                return None
        logger.info(f"Chunks file not found at {chunks_path}")
        return None

    def process_pdfs(self):
        """Process all PDFs in directory, generating or loading embeddings and indexes."""
        self.pdf_chunks = {}
        self.faiss_indexes = {}

        try:
             pdf_files = [f for f in os.listdir(self.config.pdf_dir) if f.lower().endswith(".pdf")]
        except FileNotFoundError:
             logger.error(f"PDF directory not found: {self.config.pdf_dir}")
             print(f"Error: PDF directory '{self.config.pdf_dir}' not found. Please check your config.ini.")
             return # Stop processing if directory doesn't exist
        except Exception as e:
             logger.error(f"Error listing files in PDF directory {self.config.pdf_dir}: {e}")
             print(f"Error accessing PDF directory: {e}")
             return

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.config.pdf_dir}")
            print(f"Warning: No PDF files found in '{self.config.pdf_dir}'.")
            return

        logger.info(f"Processing {len(pdf_files)} PDF files from {self.config.pdf_dir}")
        embedding_dim = self.encoder_model.get_sentence_embedding_dimension()

        for pdf_file in pdf_files:
            logger.info(f"--- Processing: {pdf_file} ---")
            pdf_path = os.path.join(self.config.pdf_dir, pdf_file)
            index_path = self.get_index_path(pdf_file)
            emb_path = self.get_embedding_path(pdf_file)
            chunks_path = self.get_chunks_path(pdf_file) # Used for checking existence

            try:
                # 1. Load or Extract/Save Chunks
                chunks = self.load_chunks(pdf_file)
                if chunks is None: # If loading failed or file doesn't exist
                    logger.info(f"Extracting and chunking content for {pdf_file}...")
                    all_content = self.extract_text_and_tables(pdf_path)
                    if not all_content:
                         logger.warning(f"No content extracted from {pdf_file}. Skipping.")
                         continue # Skip to next file if nothing extracted
                    chunks = self.chunk_content(all_content)
                    if not chunks:
                         logger.warning(f"No chunks generated for {pdf_file} after extraction. Skipping.")
                         continue # Skip if chunking yields nothing
                    self.save_chunks(pdf_file, chunks)
                self.pdf_chunks[pdf_file] = chunks

                # 2. Load or Generate/Save Embeddings and Index
                faiss_index = None
                if os.path.exists(index_path) and os.path.exists(emb_path):
                    logger.info(f"Attempting to load existing index and embeddings for {pdf_file}")
                    try:
                        embeddings = np.load(emb_path)
                        if embeddings.shape[1] != embedding_dim:
                             logger.warning(f"Embeddings dimension mismatch for {pdf_file} (Expected {embedding_dim}, Got {embeddings.shape[1]}). Regenerating.")
                             raise ValueError("Dimension mismatch") # Force regeneration

                        # Ensure number of embeddings matches number of chunks
                        if embeddings.shape[0] != len(chunks):
                             logger.warning(f"Number of embeddings ({embeddings.shape[0]}) does not match number of chunks ({len(chunks)}) for {pdf_file}. Regenerating.")
                             raise ValueError("Chunk/Embedding count mismatch") # Force regeneration

                        faiss_index = self.load_faiss_index(pdf_file, embedding_dim)
                        # Verify the index has the correct number of vectors
                        if faiss_index.ntotal != embeddings.shape[0]:
                             logger.warning(f"FAISS index count ({faiss_index.ntotal}) does not match embeddings count ({embeddings.shape[0]}) for {pdf_file}. Rebuilding index.")
                             faiss_index = faiss.IndexFlatL2(embedding_dim)
                             faiss_index.add(embeddings.astype('float32')) # Re-add embeddings
                             faiss.write_index(faiss_index, index_path)
                             logger.info(f"Rebuilt FAISS index for {pdf_file}")

                        logger.info(f"Successfully loaded embeddings ({embeddings.shape[0]} vectors) and index for {pdf_file}")

                    except Exception as e:
                        logger.error(f"Error loading pre-existing index/embeddings for {pdf_file}: {e}. Regenerating...")
                        faiss_index = None # Ensure regeneration happens

                # If index/embeddings were not loaded or need regeneration
                if faiss_index is None:
                    logger.info(f"Generating embeddings and creating index for {pdf_file}...")
                    # Prepare texts for batch encoding
                    content_list = [chunk['content'] for chunk in chunks]
                    # Batch encode for efficiency
                    embeddings = self.encoder_model.encode(
                        content_list,
                        batch_size=32, # Adjust batch size based on GPU memory
                        show_progress_bar=True,
                        convert_to_numpy=True
                    ).astype('float32')

                    if embeddings.shape[0] == 0:
                        logger.warning(f"No embeddings generated for {pdf_file}. Skipping index creation.")
                        continue

                    np.save(emb_path, embeddings)
                    logger.info(f"Saved {embeddings.shape[0]} embeddings to {emb_path}")

                    faiss_index = faiss.IndexFlatL2(embedding_dim)
                    faiss_index.add(embeddings)
                    faiss.write_index(faiss_index, index_path)
                    logger.info(f"Created and saved FAISS index with {faiss_index.ntotal} vectors to {index_path}")

                # Store the loaded/created index
                self.faiss_indexes[pdf_file] = faiss_index

            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}", exc_info=True) # Log traceback
                # Clean up potentially corrupted intermediate files for this PDF
                if os.path.exists(index_path): os.remove(index_path)
                if os.path.exists(emb_path): os.remove(emb_path)
                if os.path.exists(chunks_path): os.remove(chunks_path)
                logger.info(f"Cleaned up intermediate files for failed {pdf_file}")


    def query_pdfs(self, query):
        """Query all processed PDFs with a user query."""
        if not self.faiss_indexes:
            logger.warning("No indexes available. Run process_pdfs() first or check processing errors.")
            return {}

        all_results = {}
        try:
            logger.info(f"Encoding query: '{query[:100]}...'") # Log truncated query
            query_embedding = self.encoder_model.encode(query).astype("float32")
            # Reshape for FAISS search (expects batch dimension)
            query_embedding = np.array([query_embedding])

            logger.info(f"Searching across {len(self.faiss_indexes)} indexes.")
            for pdf_file, index in self.faiss_indexes.items():
                pdf_results = []
                # Ensure index is not empty before searching
                if index.ntotal == 0:
                    logger.warning(f"Index for {pdf_file} is empty. Skipping search.")
                    continue

                try:
                    # Determine k: min of configured k, and actual number of items in index
                    k_search = min(self.config.k_retrieval, index.ntotal)
                    logger.debug(f"Searching index for {pdf_file} with k={k_search}")

                    # D: distances (L2 squared), I: indices
                    D, I = index.search(query_embedding, k=k_search)

                    # Process search results
                    indices = I[0] # Get the array of indices for the first (only) query vector
                    distances = D[0] # Get the array of distances

                    logger.debug(f"Retrieved {len(indices)} results for {pdf_file}. Indices: {indices}, Distances: {distances}")

                    processed_indices = set() # Keep track of processed chunk indices to avoid duplicates if k > ntotal somehow
                    current_pdf_chunks = self.pdf_chunks.get(pdf_file)

                    if not current_pdf_chunks:
                         logger.error(f"Chunks for {pdf_file} not found in memory. Cannot retrieve content.")
                         continue # Skip this PDF if chunks are missing


                    for i, idx in enumerate(indices):
                        # FAISS can return -1 if k > ntotal or issues occur
                        if idx == -1 or idx in processed_indices:
                            continue

                        # Check bounds to prevent IndexError
                        if idx < 0 or idx >= len(current_pdf_chunks):
                             logger.warning(f"Retrieved invalid index {idx} for {pdf_file} (max index: {len(current_pdf_chunks)-1}). Skipping.")
                             continue

                        processed_indices.add(idx)
                        chunk = current_pdf_chunks[idx]
                        distance = distances[i]
                        # Convert L2 distance to a pseudo-confidence (higher is better)
                        # Simple inverse: 1 / (1 + distance). Needs scaling/normalization if used seriously.
                        # Or use a max distance threshold. For simplicity, just pass distance.
                        # Let's keep the original calculation for consistency, but note L2 distance means smaller is better.
                        # Confidence = 100 - distance is only meaningful if distances are somewhat normalized or bounded.
                        # Let's just report the raw distance (lower = better match).
                        score = float(distance) # Raw L2 distance


                        # Summarization is computationally expensive, consider doing it later or optionally
                        # summary = self.summarize_text(chunk['content']) # Optional: Summarize retrieved chunks

                        pdf_results.append({
                            "page": chunk.get('page', 'N/A'), # Use .get for safety
                            "content": chunk.get('content', ''),
                            # "summary": summary, # Add summary back if needed
                            "score": round(score, 4), # Lower is better (L2 distance)
                            "type": chunk.get('type', 'unknown')
                        })

                    # Sort results by score (ascending for L2 distance)
                    pdf_results.sort(key=lambda x: x['score'])
                    all_results[pdf_file] = pdf_results

                except Exception as e:
                    logger.error(f"Error searching index for {pdf_file}: {e}", exc_info=True)

            return all_results
        except Exception as e:
            logger.error(f"Error during query encoding or processing: {e}", exc_info=True)
            return {}

    def aggregate_context(self, query_results, strategy="top_k", max_context_tokens=4000):
        """
        Aggregate context from query results, respecting token limits.

        Strategies:
        - simple: Concatenate top chunks until token limit.
        - top_k: Concatenate specifically the top k chunks (defined by k_retrieval in config) if they fit.
        - weighted: (Removed for simplicity, as score interpretation is tricky)
        """
        all_context = {}
        # Rough estimate: 1 token ~ 4 chars in English
        max_chars = max_context_tokens * 3 # Leave some buffer

        if not query_results:
            logger.warning("No query results to aggregate")
            return all_context

        logger.info(f"Aggregating context using strategy: {strategy}")

        for pdf_file, results in query_results.items():
            if not results:
                logger.debug(f"No results for {pdf_file} to aggregate.")
                continue

            context = ""
            current_chars = 0

            # Results are already sorted by score (ascending L2 distance) in query_pdfs
            # So the first elements are the most relevant.

            if strategy == "simple" or strategy == "top_k":
                # top_k uses the first k results, simple uses as many as fit
                limit = self.config.k_retrieval if strategy == "top_k" else len(results)

                for i, res in enumerate(results[:limit]):
                    content_to_add = f"[Page {res['page']} - Type: {res['type']} - Score: {res['score']:.4f}]\n{res['content']}\n\n"
                    content_chars = len(content_to_add)

                    if current_chars + content_chars <= max_chars:
                        context += content_to_add
                        current_chars += content_chars
                    else:
                        # Try adding a truncated version if it's the first chunk
                        if i == 0:
                             remaining_chars = max_chars - current_chars
                             truncated_content = content_to_add[:remaining_chars]
                             context += truncated_content + "\n[...TRUNCATED...]\n\n"
                             current_chars += len(truncated_content) + len("\n[...TRUNCATED...]\n\n")
                             logger.warning(f"Context for {pdf_file} truncated to fit token limit.")
                        else:
                             logger.info(f"Stopping context aggregation for {pdf_file} due to character limit ({current_chars}/{max_chars} chars). Added {i} chunks.")
                        break # Stop adding chunks if limit reached

            # Removed 'weighted' strategy due to complexity of meaningful weighting with L2 distance.

            else:
                 logger.warning(f"Unknown aggregation strategy: {strategy}. Defaulting to 'top_k'.")
                 # Default behavior similar to top_k
                 limit = self.config.k_retrieval
                 for i, res in enumerate(results[:limit]):
                     content_to_add = f"[Page {res['page']} - Type: {res['type']} - Score: {res['score']:.4f}]\n{res['content']}\n\n"
                     content_chars = len(content_to_add)
                     if current_chars + content_chars <= max_chars:
                         context += content_to_add
                         current_chars += content_chars
                     else:
                         if i == 0:
                              remaining_chars = max_chars - current_chars
                              truncated_content = content_to_add[:remaining_chars]
                              context += truncated_content + "\n[...TRUNCATED...]\n\n"
                              current_chars += len(truncated_content) + len("\n[...TRUNCATED...]\n\n")
                              logger.warning(f"Context for {pdf_file} truncated to fit token limit.")
                         else:
                              logger.info(f"Stopping context aggregation for {pdf_file} due to character limit ({current_chars}/{max_chars} chars). Added {i} chunks.")
                         break


            final_context = context.strip()
            if final_context:
                 all_context[pdf_file] = final_context
                 logger.debug(f"Aggregated context for {pdf_file} (approx {current_chars} chars):\n{final_context[:200]}...") # Log start of context
            else:
                 logger.warning(f"No context could be aggregated for {pdf_file} within limits.")


        return all_context


    def query_llm(self, query, context, retry_count=2):
        """Query the Groq API with context and handle errors."""
        if not context:
            logger.warning("No context provided for LLM query")
            print('Error in no context')
            return self.get_fallback_response(query)

        # Get API Key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable not set.")
            print("\nError: GROQ_API_KEY environment variable is not set.")
            print("Please set the environment variable before running.")
            # Optionally: raise ValueError("GROQ_API_KEY not set.")
            return "Error: Groq API key not configured." # Return error message

        try:
            # Initialize Groq Chat Model
            llm = ChatGroq(
                temperature=self.config.temperature,
                model_name=self.config.llm_model,
                groq_api_key=api_key
            )

            # Prepare the prompt
            prompt = f"""You are an assistant specialized in answering questions based *only* on the provided context from PDF documents.
Use the following context to answer the question.

Context from document:
---
{context}
---

Question: {query}

Based *solely* on the context provided above, please answer the question.
If the context does not contain the information needed to answer the question, state clearly: "Based on the provided context, I cannot answer this question."
Do not make assumptions or use external knowledge. Be concise and precise.
Answer:"""

            logger.info(f"Querying Groq model {self.config.llm_model}...")
            # logger.debug(f"Prompt sent to Groq:\n{prompt}") # Be careful logging full prompts if they contain sensitive info
            #response = llm.invoke(prompt)
            #print(response)
            for attempt in range(retry_count + 1):
                try:
                    response = llm.invoke(prompt)
                    # Assuming response is AIMessage(content="...")
                    answer = response.content.strip()
                    logger.info(f"Groq response received (attempt {attempt+1}).")
                    # logger.debug(f"Groq raw answer: {answer}")
                    return answer

                except Exception as e:
                    logger.warning(f"Groq API query attempt {attempt+1} failed: {e}")
                    if attempt < retry_count:
                        logger.info("Retrying Groq query...")
                        # Optional: Add a small delay before retrying
                        import time
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All Groq API query attempts failed: {e}")
                        print('Error in query')
                        return self.get_fallback_response(query, error=str(e))

        except Exception as e:
            print('Error in query')
            logger.error(f"Error setting up or calling Groq API: {e}")
            return self.get_fallback_response(query, error=str(e))

    def get_fallback_response(self, query, error=None):
        """Generate a fallback response when LLM or context fails."""
        base_message = "I encountered an issue while trying to answer your question based on the documents."
        if error:
             base_message += f" (Error: {error})"

        # Simple fallback
        return base_message + " Please try rephrasing your query or check the system logs."

        # (Keeping the more nuanced fallback logic commented out for now)
        # fallback_responses = [
        #     "I couldn't find specific information about that in the documents.",
        #     "The documents don't seem to contain enough context to answer your query accurately.",
        #     "I don't have sufficient information to address your query based on the available documents.",
        #     "Your question is interesting, but I couldn't locate relevant information in the provided context."
        # ]
        # # Specific message for regulatory queries (example)
        # if re.search(r'\b\d+\s*CFR\s*\d+', query, re.IGNORECASE):
        #      specific_fallback = "I found a reference to a CFR regulation in your query, but I couldn't locate the specific regulatory details in the provided context. For accurate regulatory information, please consult the official CFR documentation or the full source document."
        #      if error:
        #           return f"{specific_fallback} (Additionally, an error occurred: {error})"
        #      return specific_fallback

        # import random
        # chosen_fallback = random.choice(fallback_responses)
        # if error:
        #      return f"{chosen_fallback} (Additionally, an error occurred: {error})"
        # return chosen_fallback


    def run_query(self, query, context_strategy="top_k"):
        """Complete query pipeline: retrieve, aggregate context, query LLM."""
        logger.info(f"--- Starting query execution for: '{query[:100]}...' ---")

        # 1. Retrieve relevant chunks from indexed PDFs
        query_results = self.query_pdfs(query)

        if not query_results:
            logger.warning("No relevant chunks found for the query across any documents.")
            # Return structure indicating no results
            return {
                "query": query,
                "results": {}, # No retrieval results
                "answers": {}, # No answers generated
                "status": "No relevant chunks found."
             }

        # 2. Aggregate context from the results
        # Using top_k by default as it's generally robust
        contexts = self.aggregate_context(query_results, strategy=context_strategy)

        if not contexts:
             logger.warning(f"Context aggregation failed or produced no content for query: '{query[:100]}...'")
             # Return structure indicating context issue
             return {
                 "query": query,
                 "results": query_results, # Include retrieval results even if context failed
                 "answers": {},
                 "status": "Failed to aggregate context from results."
             }

        # 3. Query LLM (Groq) for each document's context
        answers = {}
        for pdf_file, context in contexts.items():
            logger.info(f"Generating answer for {pdf_file}...")
            answer = self.query_llm(query, context)
            answers[pdf_file] = answer
            logger.info(f"Answer generated for {pdf_file}.")

        logger.info(f"--- Finished query execution for: '{query[:100]}...' ---")

        return {
            "query": query,
            "results": query_results, # Include raw retrieval results for inspection
            "answers": answers,
            "status": "Completed successfully."
        }


def print_results(results_data):
    """Print query results and LLM answers in a readable format."""
    query = results_data.get("query", "N/A")
    status = results_data.get("status", "Unknown status")
    answers = results_data.get("answers", {})
    retrieval_results = results_data.get("results", {})

    print("\n" + "="*60)
    print(f"Query: {query}")
    print(f"Status: {status}")
    print("="*60)


    if not answers and not retrieval_results:
        print("\nNo results or answers found.")
        print("="*60)
        return

    if answers:
        print("\n--- Generated Answers ---")
        for pdf_file, answer in answers.items():
            print(f"\n📄 Answer based on: {pdf_file}")
            print(f"{'-'*len(pdf_file)}\n{answer}")
    else:
        print("\n--- No Answers Generated ---")
        if status == "Failed to aggregate context from results.":
             print("Context could not be prepared for the LLM.")
        elif status == "No relevant chunks found.":
             print("No relevant information was found in the documents.")


    if retrieval_results:
        print("\n--- Top Retrieval Results (Lower Score is Better) ---")
        for pdf_file, pdf_res_list in retrieval_results.items():
            if pdf_res_list:
                print(f"\n📄 Top matches in: {pdf_file}")
                print(f"{'-'*len(pdf_file)}")
                for i, res in enumerate(pdf_res_list[:3]): # Show top 3 retrieved per file
                    print(f"  Rank {i+1}: Score: {res['score']:.4f} (Page {res['page']}, Type: {res['type']})")
                    # Truncate content preview
                    content_preview = res['content'].strip().replace('\n', ' ')[:150]
                    print(f"     Content: {content_preview}...")
            # else: # Optional: Mention files with no matches
            #     print(f"\n📄 No relevant chunks found in: {pdf_file}")
    else:
         # This case should be covered by the initial check, but added for completeness
         print("\n--- No Retrieval Results Found ---")

    print("\n" + "="*60)



def main():
    """Main function to run the RAG system."""
    try:
        # Check for API key early
        if not os.getenv("GROQ_API_KEY"):
             print("\nError: GROQ_API_KEY environment variable is not set.")
             print("Please set the environment variable before running the script.")
             print("Example (Linux/macOS): export GROQ_API_KEY='your_key_here'")
             print("Example (Windows CMD): set GROQ_API_KEY=your_key_here")
             print("Example (Windows PowerShell): $env:GROQ_API_KEY = 'your_key_here'")
             return # Exit if key is not set

        # Initialize and process PDFs
        rag = RAGSystem() # Consider passing config path if not default
        rag.process_pdfs()

        # Check if any indexes were actually loaded/created
        if not rag.faiss_indexes:
             print("\nNo PDF documents were successfully processed or indexed.")
             print(f"Please check the '{rag.config.pdf_dir}' directory for PDFs and review the 'rag_system.log' file for errors.")
             return # Exit if no indexes are ready

        print("\n--- RAG System Ready ---")
        print(f"Using Groq model: {rag.config.llm_model}")
        print(f"Indexed {len(rag.faiss_indexes)} PDF document(s).")
        print("--- Enter your query below ---")

        # Interactive query loop
        while True:
            try:
                 query = input("\nEnter query (or 'exit' to quit): ")
                 if query.lower().strip() in ["exit", "quit", "q"]:
                     break
                 if not query.strip():
                      continue # Ignore empty input

                 results_data = rag.run_query(query, context_strategy="top_k") # Use top_k context
                 print_results(results_data)

            except EOFError: # Handle Ctrl+D
                 print("\nExiting...")
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 print("\nExiting...")
                 break

    except KeyboardInterrupt:
        print("\nOperation cancelled by user during startup.")
    except FileNotFoundError as e:
         logger.error(f"Initialization failed: {e}")
         print(f"\nError: A required file or directory was not found: {e}")
         print("Please ensure configuration paths in config.ini are correct.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main function: {e}", exc_info=True) # Log traceback
        print(f"\nAn unexpected error occurred: {e}")
        print("Check 'rag_system.log' for more details.")


if __name__ == "__main__":
    main()
# --- END OF FILE RAG2.py ---