"""
Advanced AI Agent Framework with enhanced security, learning capabilities, and modular design.
"""

import os
import sys
import json
import logging
import logging.handlers  # Added missing import for RotatingFileHandler
import datetime
import uuid
import tempfile
import subprocess
import threading
import re
import signal
import hashlib
import sqlite3
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Core data processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For deep learning (with conditional imports)
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging with rotation
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file rotation"""
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also add console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
logger = setup_logger("AI_Agent", "logs/ai_agent.log")

class Singleton(type):
    """Metaclass for implementing the Singleton pattern."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        """Ensure only one instance of a class exists."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigManager(metaclass=Singleton):
    """Manages configuration settings for the AI agent."""
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            
        # Default configuration
        return {
            "model": {
                "embedding_model": "all-MiniLM-L6-v2",
                "generation_model": "gpt2",
                "use_gpu": True,
                "model_cache_dir": ".model_cache"
            },
            "security": {
                "safe_mode": True,
                "execution_timeout": 10,  # seconds
                "max_memory_usage": 512,  # MB
                "allowed_modules": ["math", "random", "datetime", "collections", "re", "json"],
                "blocked_modules": ["os", "sys", "subprocess", "shutil", "socket"]
            },
            "learning": {
                "max_history": 1000,
                "feedback_threshold": 0.7,
                "auto_update_interval": 100  # interactions
            },
            "database": {
                "path": "agent_data.db",
                "vector_dimension": 384
            },
            "api": {
                "enabled": False,
                "port": 5000,
                "allowed_origins": ["localhost"]
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, section, key=None):
        """Get a configuration value."""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set a configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        return self.save_config()

class VectorDatabase:
    """Manages a vector database for storing and retrieving embeddings."""
    
    def __init__(self, db_path="agent_data.db", vector_dim=384):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create vectors table with appropriate number of dimensions
            vector_columns = ", ".join([f"dim_{i} REAL" for i in range(self.vector_dim)])
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS vectors (
                embedding_id TEXT PRIMARY KEY,
                {vector_columns},
                FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
    
    def add_embedding(self, text, embedding, source="user_interaction", tags=None):
        """Add a text and its embedding to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate a unique ID
            embedding_id = str(uuid.uuid4())
            
            # Add text entry
            cursor.execute(
                "INSERT INTO embeddings (id, text, source, tags) VALUES (?, ?, ?, ?)",
                (embedding_id, text, source, json.dumps(tags or []))
            )
            
            # Add vector entry - ensure embedding has correct dimensions
            if len(embedding) != self.vector_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.vector_dim}, got {len(embedding)}")
            
            placeholders = ", ".join(["?"] * (self.vector_dim + 1))
            values = [embedding_id] + list(embedding)
            
            columns = ["embedding_id"] + [f"dim_{i}" for i in range(self.vector_dim)]
            columns_str = ", ".join(columns)
            
            cursor.execute(f"INSERT INTO vectors ({columns_str}) VALUES ({placeholders})", values)
            
            conn.commit()
            conn.close()
            return embedding_id
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return None
    
    def search_similar(self, embedding, limit=5, threshold=0.7):
        """Find similar embeddings using cosine similarity."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # We'll use a simplified approach for SQLite by calculating distances
            # For production, consider using specialized vector databases like Milvus, FAISS, or Pinecone
            
            # Get all vectors
            columns = [f"dim_{i}" for i in range(self.vector_dim)]
            columns_str = ", ".join(columns)
            cursor.execute(f"SELECT embedding_id, {columns_str} FROM vectors")
            
            results = []
            query_embedding = np.array(embedding).reshape(1, -1)
            
            for row in cursor.fetchall():
                embedding_id = row[0]
                vector = np.array(row[1:]).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, vector)[0][0]
                
                if similarity >= threshold:
                    results.append((embedding_id, similarity))
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
            
            # Get texts for the similar embeddings
            similar_items = []
            for embedding_id, similarity in results:
                cursor.execute("SELECT text, source, tags FROM embeddings WHERE id = ?", (embedding_id,))
                text, source, tags = cursor.fetchone()
                similar_items.append({
                    "id": embedding_id,
                    "text": text,
                    "similarity": similarity,
                    "source": source,
                    "tags": json.loads(tags)
                })
            
            conn.close()
            return similar_items
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            return []
    
    def delete_embedding(self, embedding_id):
        """Delete an embedding from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM vectors WHERE embedding_id = ?", (embedding_id,))
            cursor.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return False

class ModelManager:
    """Manages AI models for embeddings and text generation."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or ConfigManager()
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        
        # Set cache directory for models
        os.environ['TRANSFORMERS_CACHE'] = self.config.get("model", "model_cache_dir")
        os.makedirs(self.config.get("model", "model_cache_dir"), exist_ok=True)
        
        self.device = self._setup_device()
        
    def _setup_device(self):
        """Set up the appropriate device for model inference."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, using CPU fallback")
            return "cpu"
            
        if self.config.get("model", "use_gpu") and torch.cuda.is_available():
            logger.info("Using GPU for model inference")
            return "cuda"
        else:
            logger.info("Using CPU for model inference")
            return "cpu"
    
    def load_embedding_model(self, force_reload=False):
        """Load the text embedding model."""
        if self.embedding_model is not None and not force_reload:
            return self.embedding_model
            
        if not HAS_TORCH:
            logger.warning("PyTorch not available, using TF-IDF fallback for embeddings")
            self.embedding_model = TfidfVectorizer()
            return self.embedding_model
            
        try:
            model_name = self.config.get("model", "embedding_model")
            logger.info(f"Loading embedding model: {model_name}")
            
            # Use sentence-transformers for embeddings if available
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model.to(self.device)
            
            logger.info("Embedding model loaded successfully")
            return self.embedding_model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Using TF-IDF fallback for embeddings")
            self.embedding_model = TfidfVectorizer()
            return self.embedding_model
    
    def load_generation_model(self, force_reload=False):
        """Load the text generation model."""
        if self.generation_model is not None and not force_reload:
            return self.generation_model, self.tokenizer
            
        if not HAS_TORCH:
            logger.warning("PyTorch not available, cannot load generation model")
            return None, None
            
        try:
            model_name = self.config.get("model", "generation_model")
            logger.info(f"Loading generation model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # Add low memory options for smaller systems
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.generation_model.to(self.device)
            
            logger.info("Generation model loaded successfully")
            return self.generation_model, self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            return None, None
    
    def get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        model = self.load_embedding_model()
        
        try:
            if isinstance(model, TfidfVectorizer):
                # If using TF-IDF fallback
                if not hasattr(model, 'vocabulary_') or not model.vocabulary_:
                    model.fit(texts)
                vectors = model.transform(texts).toarray()
                # Normalize vectors for consistent similarity calculation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-10)
                return vectors
            else:
                # Using sentence-transformers
                return model.encode(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), 
                             self.config.get("database", "vector_dimension")))
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text using the loaded language model."""
        model, tokenizer = self.load_generation_model()
        
        if model is None or tokenizer is None:
            return "Text generation model not available."
            
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.# This workflow will install Python dependencies, run tests and lint with a single version of Python
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

name: Python application

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

permissions:
  contents: read

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.10
      uses: actions/setup-python@v3
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        pytest
