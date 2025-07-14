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
from langchain_community.llms import Ollama

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
                "llm_model": "mistral",
                "device": "auto"  # auto, cuda, or cpu
            },
            "PARAMETERS": {
                "chunk_size": "200",
                "overlap": "50",
                "k_retrieval": "5",
                "temperature": "0.2"
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
        return self.config.get("MODELS", "llm_model")
    
    @property
    def device(self):
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
        return self.config.getfloat("PARAMETERS", "temperature")


class RAGSystem:
    """Retrieval-Augmented Generation system for PDF documents."""
    
    def __init__(self, config_path="config.ini"):
        """Initialize the RAG system with configuration."""
        self.config = RAGConfig(config_path)
        logger.info(f"Initializing RAG system with device: {self.config.device}")
        
        # Create necessary directories
        os.makedirs(self.config.pdf_dir, exist_ok=True)
        os.makedirs(self.config.index_dir, exist_ok=True)
        
        # Initialize models
        try:
            self.encoder_model = SentenceTransformer(self.config.encoder_model).to(self.config.device)
            self.summarizer_model = T5ForConditionalGeneration.from_pretrained(self.config.summarizer_model).to(self.config.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.summarizer_model)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to initialize models: {e}")
        
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
                            table_string = table_df.to_string()
                            all_content.append({
                                "type": "table", 
                                "content": table_string, 
                                "page": page_num + 1
                            })
                        except Exception as e:
                            logger.warning(f"Error processing table on page {page_num + 1}: {e}")
            
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
            words = content.split()
            
            if not words:
                continue
                
            for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
                chunk = " ".join(words[i:i + self.config.chunk_size])
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
            input_text = "summarize: " + text
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.config.device)
            summary_ids = self.summarizer_model.generate(
                input_ids, 
                max_length=150, 
                num_beams=4, 
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return "Failed to generate summary"
    
    def get_index_path(self, pdf_file):
        """Get path for FAISS index file."""
        return os.path.join(self.config.index_dir, f"{pdf_file.replace('.pdf', '')}.index")
    
    def get_embedding_path(self, pdf_file):
        """Get path for embedding numpy file."""
        return os.path.join(self.config.index_dir, f"{pdf_file.replace('.pdf', '')}.npy")
    
    def get_chunks_path(self, pdf_file):
        """Get path for chunks JSON file."""
        return os.path.join(self.config.index_dir, f"{pdf_file.replace('.pdf', '')}.json")
    
    def load_faiss_index(self, pdf_file, embedding_dim):
        """Load FAISS index from disk or create new one."""
        index_path = self.get_index_path(pdf_file)
        if os.path.exists(index_path):
            try:
                return faiss.read_index(index_path)
            except Exception as e:
                logger.error(f"Error reading FAISS index for {pdf_file}: {e}")
        
        # Create new index
        return faiss.IndexFlatL2(embedding_dim)
    
    def save_chunks(self, pdf_file, chunks):
        """Save chunks to disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        try:
            with open(chunks_path, 'w') as f:
                json.dump(chunks, f)
            logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        except Exception as e:
            logger.error(f"Error saving chunks for {pdf_file}: {e}")
    
    def load_chunks(self, pdf_file):
        """Load chunks from disk."""
        chunks_path = self.get_chunks_path(pdf_file)
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, 'r') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
                return chunks
            except Exception as e:
                logger.error(f"Error loading chunks for {pdf_file}: {e}")
        return None
    
    def process_pdfs(self):
        """Process all PDFs in directory."""
        self.pdf_chunks = {}
        self.faiss_indexes = {}
        
        pdf_files = [f for f in os.listdir(self.config.pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.config.pdf_dir}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.config.pdf_dir, pdf_file)
                emb_path = self.get_embedding_path(pdf_file)
                index_path = self.get_index_path(pdf_file)
                
                # Try to load chunks from disk first
                chunks = self.load_chunks(pdf_file)
                if not chunks:
                    all_content = self.extract_text_and_tables(pdf_path)
                    chunks = self.chunk_content(all_content)
                    self.save_chunks(pdf_file, chunks)
                
                if not chunks:
                    logger.warning(f"No content extracted from {pdf_file}")
                    continue
                
                self.pdf_chunks[pdf_file] = chunks
                
                # Load or create embeddings and index
                if os.path.exists(emb_path) and os.path.exists(index_path):
                    try:
                        embeddings = np.load(emb_path)
                        index = self.load_faiss_index(pdf_file, embeddings.shape[1])
                        logger.info(f"Loaded existing embeddings and index for {pdf_file}")
                    except Exception as e:
                        logger.error(f"Error loading embeddings for {pdf_file}: {e}")
                        # Regenerate if loading fails
                        embeddings = np.array([
                            self.encoder_model.encode(chunk['content']) 
                            for chunk in chunks
                        ]).astype("float32")
                        np.save(emb_path, embeddings)
                        index = faiss.IndexFlatL2(embeddings.shape[1])
                        index.add(embeddings)
                        faiss.write_index(index, index_path)
                        logger.info(f"Regenerated embeddings and index for {pdf_file}")
                else:
                    embeddings = np.array([
                        self.encoder_model.encode(chunk['content']) 
                        for chunk in chunks
                    ]).astype("float32")
                    np.save(emb_path, embeddings)
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings)
                    faiss.write_index(index, index_path)
                    logger.info(f"Created embeddings and index for {pdf_file}")
                
                self.faiss_indexes[pdf_file] = index
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
    
    def query_pdfs(self, query):
        """Query all PDFs with a user query."""
        if not self.faiss_indexes:
            logger.warning("No indexes available. Run process_pdfs() first.")
            return {}
        
        results = {}
        try:
            query_embedding = self.encoder_model.encode(query).astype("float32")
            
            for pdf, index in self.faiss_indexes.items():
                pdf_results = []
                
                try:
                    D, I = index.search(np.array([query_embedding]), k=min(self.config.k_retrieval, index.ntotal))
                    
                    for i, idx in enumerate(I[0]):
                        if idx < len(self.pdf_chunks[pdf]):
                            chunk = self.pdf_chunks[pdf][idx]
                            summary = self.summarize_text(chunk['content'])
                            
                            pdf_results.append({
                                "page": chunk['page'],
                                "content": chunk['content'],
                                "summary": summary,
                                "confidence": round(100 - D[0][i], 2),
                                "type": chunk['type']
                            })
                    
                    results[pdf] = pdf_results
                except Exception as e:
                    logger.error(f"Error querying {pdf}: {e}")
            
            return results
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {}
    
    def aggregate_context(self, query_results, strategy="weighted"):
        """
        Aggregate context from query results.
        
        Strategies:
        - simple: concatenate all chunks
        - weighted: weight chunks by confidence score
        - top_k: only use top k chunks
        """
        all_context = {}
        
        if not query_results:
            logger.warning("No query results to aggregate")
            return all_context
        
        for pdf_file, results in query_results.items():
            if not results:
                continue
                
            context = ""
            
            if strategy == "simple":
                # Simple concatenation
                for res in results:
                    context += f"[Page {res['page']} - {res['type']}]: {res['content']}\n\n"
            
            elif strategy == "weighted":
                # Weight by confidence score
                sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
                for res in sorted_results:
                    weight = res['confidence'] / 100.0  # Normalize to 0-1
                    # Include more content from higher confidence chunks
                    content_length = int(len(res['content']) * weight)
                    content = res['content'][:content_length] if content_length > 0 else res['summary']
                    context += f"[Page {res['page']} - {res['type']} - Confidence: {res['confidence']}%]: {content}\n\n"
            
            elif strategy == "top_k":
                # Use only top k chunks
                k = min(3, len(results))  # Use top 3 or fewer
                sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:k]
                for res in sorted_results:
                    context += f"[Page {res['page']} - {res['type']}]: {res['content']}\n\n"
            
            all_context[pdf_file] = context.strip()
        
        return all_context
    
    def query_llm(self, query, context, retry_count=2):
        """Query the LLM with context and handle errors."""
        if not context:
            logger.warning("No context provided for LLM query")
            return self.get_fallback_response(query)
        
        try:
            llm = Ollama(model=self.config.llm_model, temperature=self.config.temperature)
            prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer the question based on the provided context. If the context doesn't contain relevant information, state that clearly."
            
            for attempt in range(retry_count + 1):
                try:
                    answer = llm(prompt)
                    return answer
                except Exception as e:
                    if attempt < retry_count:
                        logger.warning(f"LLM query attempt {attempt+1} failed: {e}. Retrying...")
                        continue
                    else:
                        logger.error(f"All LLM query attempts failed: {e}")
                        return self.get_fallback_response(query)
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            return self.get_fallback_response(query)
    
    def get_fallback_response(self, query):
        """Generate a fallback response when LLM or context fails."""
        fallback_responses = [
            "I couldn't find specific information about that in the documents.",
            "The documents don't contain enough context to answer your query accurately.",
            "I don't have sufficient information to address your query based on the available documents.",
            "Your question is interesting, but I couldn't locate relevant information in the documents."
        ]
        
        # Return a specific message for regulatory or technical queries
        if re.search(r'\b\d+\s*CFR\s*\d+', query):
            return "I found a reference to a CFR regulation in your query, but I couldn't locate the specific regulatory details in the documents. For accurate regulatory information, please consult the official CFR documentation."
        
        import random
        return random.choice(fallback_responses)
    
    def run_query(self, query, context_strategy="weighted"):
        """Complete query pipeline."""
        logger.info(f"Processing query: {query}")
        
        # Get raw retrieval results
        query_results = self.query_pdfs(query)
        
        if not query_results:
            logger.warning("No results found for query")
            return {"results": {}, "answers": {}}
        
        # Aggregate context
        contexts = self.aggregate_context(query_results, strategy=context_strategy)
        
        # Query LLM for each document
        answers = {}
        for pdf_file, context in contexts.items():
            answer = self.query_llm(query, context)
            answers[pdf_file] = answer
        
        return {
            "results": query_results,
            "answers": answers
        }


def print_results(results_data):
    """Print query results and LLM answers in a readable format."""
    if not results_data or not results_data.get("results"):
        print("\nNo results found.")
        return
    
    print("\n--- Query Results ---")
    
    for pdf_file, answer in results_data.get("answers", {}).items():
        print(f"\n📄 {pdf_file}")
        print(f"Ollama's Answer:\n{answer}")
        


def main():
    """Main function to run the RAG system."""
    try:
        # Initialize and process PDFs
        rag = RAGSystem()
        rag.process_pdfs()
        
        # Interactive query loop
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit", "q"]:
                break
                
            results = rag.run_query(query, context_strategy="weighted")
            print_results(results)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()