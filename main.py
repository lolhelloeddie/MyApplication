import os
import json
import datetime
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from cryptography.fernet import Fernet

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

class AIAssistant:
    """
    An advanced AI assistant that learns from interactions and specializes in coding,
    cybersecurity, and machine learning.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the AI assistant with configuration settings."""
        self.config = self._load_config(config_path)
        self.name = self.config.get("assistant_name", "Athena")
        self.version = self.config.get("version", "0.1.0")
        
        # Set up encryption for sensitive data
        self.encryption_key = self._get_or_create_key()
        self.encryptor = Fernet(self.encryption_key)
        
        # Initialize language model and embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_id"])
        self.model = AutoModel.from_pretrained(self.config["model_id"])
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings_model"]
        )
        
        # Set up memory systems
        self.short_term_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.long_term_memory = self._load_or_create_long_term_memory()
        
        # Set up knowledge base
        self.knowledge_base = self._load_or_create_knowledge_base()
        
        # Initialize learning components
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.user_preferences = self._load_user_preferences()
        self.interaction_history = []
        
        # Specialized modules
        self.coding_module = CodingModule(self.config.get("coding_config", {}))
        self.cybersecurity_module = CybersecurityModule(self.config.get("security_config", {}))
        self.ml_module = MachineLearningModule(self.config.get("ml_config", {}))
        
        print(f"{self.name} v{self.version} initialized successfully.")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default configuration
            default_config = {
                "assistant_name": "Athena",
                "version": "0.1.0",
                "model_id": "sentence-transformers/all-mpnet-base-v2",
                "embeddings_model": "sentence-transformers/all-mpnet-base-v2",
                "openai_api_key": "",  # To be filled by user
                "learning_rate": 0.01,
                "memory_path": "memory/",
                "knowledge_base_path": "knowledge/",
                "coding_config": {
                    "preferred_languages": ["python", "javascript", "rust"],
                    "code_repositories": []
                },
                "security_config": {
                    "scan_frequency": "daily",
                    "vulnerability_databases": ["CVE", "OWASP"]
                },
                "ml_config": {
                    "frameworks": ["pytorch", "tensorflow", "scikit-learn"],
                    "model_storage_path": "models/"
                }
            }
            
            # Ensure directories exist
            os.makedirs("memory", exist_ok=True)
            os.makedirs("knowledge", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            return default_config
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        key_path = "secret.key"
        try:
            with open(key_path, 'rb') as f:
                key = f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
        return key
    
    def _load_or_create_long_term_memory(self) -> FAISS:
        """Load existing vector store or create a new one."""
        memory_path = self.config["memory_path"]
        try:
            return FAISS.load_local(memory_path, self.embeddings)
        except Exception:
            # Initialize empty vector store
            empty_texts = ["Initial memory placeholder"]
            embeddings = self.embeddings.embed_documents(empty_texts)
            return FAISS.from_embeddings(
                text_embeddings=zip(empty_texts, embeddings),
                embedding=self.embeddings
            )
    
    def _load_or_create_knowledge_base(self) -> FAISS:
        """Load existing knowledge base or create a new one."""
        kb_path = self.config["knowledge_base_path"]
        try:
            return FAISS.load_local(kb_path, self.embeddings)
        except Exception:
            # Initialize with some basic knowledge
            basic_knowledge = [
                "Python is a high-level, interpreted programming language.",
                "Cybersecurity involves protecting systems from digital attacks.",
                "Machine learning is a subset of artificial intelligence."
            ]
            embeddings = self.embeddings.embed_documents(basic_knowledge)
            return FAISS.from_embeddings(
                text_embeddings=zip(basic_knowledge, embeddings),
                embedding=self.embeddings
            )
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences or create defaults."""
        pref_path = "user_preferences.json"
        try:
            with open(pref_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            default_prefs = {
                "preferred_response_length": "medium",
                "code_style": "pep8",
                "interests": [],
                "feedback_history": []
            }
            with open(pref_path, 'w') as f:
                json.dump(default_prefs, f, indent=4)
            return default_prefs
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        This is the main interaction method.
        """
        # Record the interaction
        timestamp = datetime.datetime.now().isoformat()
        self.interaction_history.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "processed": False
        })
        
        # Analyze input to determine intent and domain
        intent, domain = self._analyze_input(user_input)
        
        # Get relevant context from memory and knowledge base
        context = self._retrieve_context(user_input, domain)
        
        # Generate response based on domain
        if domain == "coding":
            response = self.coding_module.generate_response(user_input, context)
        elif domain == "cybersecurity":
            response = self.cybersecurity_module.generate_response(user_input, context)
        elif domain == "machine_learning":
            response = self.ml_module.generate_response(user_input, context)
        else:
            # General response
            response = self._generate_general_response(user_input, context)
        
        # Update memory with this interaction
        self._update_memory(user_input, response)
        
        # Learn from the interaction
        self._learn_from_interaction(user_input, response)
        
        # Update the interaction record
        self.interaction_history[-1]["processed"] = True
        self.interaction_history[-1]["response"] = response
        self.interaction_history[-1]["domain"] = domain
        
        return response
    
    def _analyze_input(self, text: str) -> Tuple[str, str]:
        """
        Analyze user input to determine intent and domain.
        Returns a tuple of (intent, domain).
        """
        # Simple keyword-based approach - in a real system, use a classifier
        coding_keywords = ["code", "programming", "function", "class", "bug", "error", "syntax"]
        security_keywords = ["security", "vulnerability", "hack", "encryption", "firewall", "exploit"]
        ml_keywords = ["machine learning", "model", "train", "predict", "classify", "cluster", "neural"]
        
        # Count matches in each domain
        coding_score = sum(1 for kw in coding_keywords if kw.lower() in text.lower())
        security_score = sum(1 for kw in security_keywords if kw.lower() in text.lower())
        ml_score = sum(1 for kw in ml_keywords if kw.lower() in text.lower())
        
        # Determine domain based on highest score
        scores = [
            ("coding", coding_score),
            ("cybersecurity", security_score),
            ("machine_learning", ml_score)
        ]
        domain = max(scores, key=lambda x: x[1])[0] if max(scores, key=lambda x: x[1])[1] > 0 else "general"
        
        # Simple intent detection
        intents = {
            "question": ["what", "how", "why", "when", "where", "who"],
            "command": ["create", "make", "build", "generate", "analyze"],
            "information": ["tell", "explain", "describe", "elaborate"]
        }
        
        intent = "general"
        for intent_type, indicators in intents.items():
            if any(text.lower().startswith(ind) for ind in indicators):
                intent = intent_type
                break
                
        return intent, domain
    
    def _retrieve_context(self, query: str, domain: str) -> List[str]:
        """
        Retrieve relevant context from memory and knowledge base.
        """
        # Convert query to embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in long-term memory
        memory_results = self.long_term_memory.similarity_search_by_vector(
            query_embedding, 
            k=3
        )
        
        # Search in knowledge base
        kb_results = self.knowledge_base.similarity_search_by_vector(
            query_embedding,
            k=5
        )
        
        # Combine results
        context = [doc.page_content for doc in memory_results + kb_results]
        
        # Add domain-specific context
        if domain == "coding":
            context.extend(self.coding_module.get_relevant_knowledge(query))
        elif domain == "cybersecurity":
            context.extend(self.cybersecurity_module.get_relevant_knowledge(query))
        elif domain == "machine_learning":
            context.extend(self.ml_module.get_relevant_knowledge(query))
            
        return context
    
    def _generate_general_response(self, query: str, context: List[str]) -> str:
        """Generate a general response when no specific domain applies."""
        # In a real implementation, use a more sophisticated approach
        # This is a simple placeholder
        response = f"I understand you're asking about: {query}\n\n"
        
        if context:
            response += "Based on what I know, I can tell you that:\n"
            for i, ctx in enumerate(context[:3], 1):
                response += f"{i}. {ctx}\n"
        else:
            response += "I don't have specific information about this yet. "
            response += "Would you like me to learn more about this topic?"
            
        return response
    
    def _update_memory(self, user_input: str, response: str) -> None:
        """Update the assistant's memory with the current interaction."""
        # Add to short-term memory
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        # Add to long-term memory
        interaction_text = f"User: {user_input}\nAssistant: {response}"
        self.long_term_memory.add_texts([interaction_text])
        
        # Periodically save long-term memory
        if len(self.interaction_history) % 10 == 0:
            self.save_memory()
    
    def _learn_from_interaction(self, user_input: str, response: str) -> None:
        """Learn from the current interaction to improve future responses."""
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(user_input)
        
        # Update user preferences based on interaction
        # In a real system, use more sophisticated preference learning
        if sentiment["compound"] > 0.5:
            # Very positive reaction - note this as a good response pattern
            self.user_preferences["feedback_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "pattern": "positive",
                "interaction": len(self.interaction_history) - 1
            })
            
        # Extract keywords from this interaction for future reference
        # This is a simplified approach - real systems would use more advanced techniques
        words = user_input.lower().split()
        stopwords = nltk.corpus.stopwords.words('english')
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        if keywords:
            if "interests" not in self.user_preferences:
                self.user_preferences["interests"] = []
            
            for kw in keywords:
                if kw not in self.user_preferences["interests"]:
                    self.user_preferences["interests"].append(kw)
    
    def add_to_knowledge_base(self, text: str, source: str = "user") -> None:
        """Add new information to the knowledge base."""
        # Add metadata about source and timestamp
        metadata = {
            "source": source,
            "added": datetime.datetime.now().isoformat()
        }
        
        # Split text into chunks if it's long
        if len(text) > 1000:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(text)
        else:
            chunks = [text]
            
        # Add to knowledge base
        for chunk in chunks:
            self.knowledge_base.add_texts([chunk], metadatas=[metadata])
    
    def save_memory(self) -> None:
        """Save the assistant's memory to disk."""
        memory_path = self.config["memory_path"]
        self.long_term_memory.save_local(memory_path)
        
        # Save user preferences
        with open("user_preferences.json", 'w') as f:
            json.dump(self.user_preferences, f, indent=4)
            
        # Save interaction history (encrypted)
        history_bytes = json.dumps(self.interaction_history).encode()
        encrypted_history = self.encryptor.encrypt(history_bytes)
        with open("interaction_history.enc", 'wb') as f:
            f.write(encrypted_history)
            
        print(f"Memory saved at {datetime.datetime.now().isoformat()}")
    
    def load_memory(self) -> None:
        """Load the assistant's memory from disk."""
        try:
            # Load interaction history
            with open("interaction_history.enc", 'rb') as f:
                encrypted_history = f.read()
            history_bytes = self.encryptor.decrypt(encrypted_history)
            self.interaction_history = json.loads(history_bytes.decode())
            
            # Load user preferences
            with open("user_preferences.json", 'r') as f:
                self.user_preferences = json.load(f)
                
            print(f"Memory loaded successfully. {len(self.interaction_history)} previous interactions found.")
        except FileNotFoundError:
            print("No previous memory found.")
            
    def train_from_data(self, data_path: str) -> None:
        """Train the assistant on new data."""
        if not os.path.exists(data_path):
            print(f"Data path {data_path} not found.")
            return
            
        try:
            # Load data from directory
            loader = DirectoryLoader(data_path)
            documents = loader.load()
            
            # Process and add to knowledge base
            for doc in documents:
                self.add_to_knowledge_base(doc.page_content, source=doc.metadata.get("source", "data_import"))
                
            print(f"Successfully trained on {len(documents)} documents from {data_path}")
        except Exception as e:
            print(f"Error training from data: {str(e)}")


class CodingModule:
    """Module specialized in programming and software development."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preferred_languages = config.get("preferred_languages", ["python"])
        self.repositories = config.get("code_repositories", [])
        
        # Initialize code knowledge base
        self.code_knowledge = {}
        for lang in self.preferred_languages:
            self._load_language_knowledge(lang)
    
    def _load_language_knowledge(self, language: str) -> None:
        """Load knowledge about a specific programming language."""
        knowledge_file = f"knowledge/lang_{language}.json"
        
        try:
            with open(knowledge_file, 'r') as f:
                self.code_knowledge[language] = json.load(f)
        except FileNotFoundError:
            # Create basic knowledge for language
            self.code_knowledge[language] = {
                "syntax_examples": {},
                "common_libraries": {},
                "best_practices": [],
                "common_errors": {}
            }
            
            # Save the basic knowledge
            with open(knowledge_file, 'w') as f:
                json.dump(self.code_knowledge[language], f, indent=4)
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response to a coding-related query."""
        # Detect programming language in query
        language = self._detect_language(query)
        
        # Check if it's a code generation request
        if any(kw in query.lower() for kw in ["write", "create", "generate", "code", "function"]):
            return self._generate_code(query, language, context)
        
        # Check if it's a debugging request
        if any(kw in query.lower() for kw in ["fix", "debug", "error", "issue", "problem"]):
            return self._debug_code(query, language, context)
        
        # General coding question
        return self._answer_coding_question(query, language, context)
    
    def _detect_language(self, text: str) -> str:
        """Detect the programming language being discussed."""
        for lang in self.preferred_languages:
            if lang.lower() in text.lower():
                return lang
        return self.preferred_languages[0]  # Default to first preferred language
    
    def _generate_code(self, query: str, language: str, context: List[str]) -> str:
        """Generate code based on user query."""
        # In a production system, this would call a specialized code generation model
        # Here's a simplified implementation
        
        response = f"Here's a {language} implementation that addresses your request:\n\n"
        
        if language == "python":
            # Very simple code generation for demonstration
            if "function" in query.lower():
                function_name = "example_function"
                response += f"```python\ndef {function_name}(parameter1, parameter2):\n    \"\"\"\n    This function demonstrates the implementation requested.\n    \n    Args:\n        parameter1: First parameter description\n        parameter2: Second parameter description\n        \n    Returns:\n        Description of what the function returns\n    \"\"\"\n    # Implementation\n    result = parameter1 + parameter2\n    \n    return result\n```\n\n"
            else:
                response += "```python\n# Example implementation\n\nimport pandas as pd\nimport numpy as np\n\n# Load and prepare data\ndata = pd.read_csv('your_data.csv')\n\n# Process data\nprocessed_data = data.dropna()\nresult = processed_data.groupby('category').mean()\n\nprint(result)\n```\n\n"
        
        response += "Would you like me to explain this code or modify it in any way?"
        return response
    
    def _debug_code(self, query: str, language: str, context: List[str]) -> str:
        """Help debug code issues."""
        # In a real system, this would analyze provided code snippets
        # This is a simplified version
        
        response = "Based on your description, here are potential issues and solutions:\n\n"
        
        if "syntax error" in query.lower():
            response += "1. **Syntax Error**: Check for missing parentheses, brackets, or quotation marks.\n"
            response += "2. Ensure all blocks (if statements, loops, functions) are properly closed.\n"
            response += "3. Look for missing colons at the end of conditional statements or function definitions.\n\n"
        
        if "runtime error" in query.lower() or "exception" in query.lower():
            response += "1. **Runtime Error**: Make sure all variables are defined before use.\n"
            response += "2. Check for division by zero or accessing invalid index positions.\n"
            response += "3. Ensure proper type conversion when working with different data types.\n\n"
        
        response += "If you share the specific error message or code snippet, I can provide more targeted debugging assistance."
        return response
    
    def _answer_coding_question(self, query: str, language: str, context: List[str]) -> str:
        """Answer general coding questions."""
        # Use context to provide relevant information
        if context:
            relevant_info = "\n".join(context[:3])
            response = f"Regarding your coding question about {language}:\n\n{relevant_info}\n\n"
        else:
            response = f"Regarding your question about {language} programming:\n\n"
        
        # Add language-specific knowledge if available
        if language in self.code_knowledge:
            if "best_practices" in self.code_knowledge[language]:
                practices = self.code_knowledge[language]["best_practices"]
                if practices:
                    response += "Here are some best practices that might be relevant:\n"
                    for practice in practices[:3]:
                        response += f"- {practice}\n"
        
        return response
    
    def get_relevant_knowledge(self, query: str) -> List[str]:
        """Get relevant coding knowledge for a query."""
        language = self._detect_language(query)
        
        knowledge = []
        if language in self.code_knowledge:
            # Add some basic language facts
            if "syntax_examples" in self.code_knowledge[language]:
                for syntax, example in list(self.code_knowledge[language]["syntax_examples"].items())[:2]:
                    knowledge.append(f"{language} {syntax}: {example}")
                    
            # Add best practices
            if "best_practices" in self.code_knowledge[language]:
                for practice in self.code_knowledge[language]["best_practices"][:2]:
                    knowledge.append(f"Best practice in {language}: {practice}")
                    
        return knowledge


class CybersecurityModule:
    """Module specialized in cybersecurity."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vulnerability_dbs = config.get("vulnerability_databases", ["CVE"])
        self.scan_frequency = config.get("scan_frequency", "weekly")
        
        # Load security knowledge base
        self.security_knowledge = self._load_security_knowledge()
    
    def _load_security_knowledge(self) -> Dict[str, Any]:
        """Load cybersecurity knowledge."""
        knowledge_file = "knowledge/security_knowledge.json"
        
        try:
            with open(knowledge_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create basic security knowledge
            basic_knowledge = {
                "principles": [
                    "Defense in depth - Use multiple layers of security controls",
                    "Principle of least privilege - Limit access rights to the minimum necessary",
                    "Keep it simple - Complexity increases security risks"
                ],
                "common_vulnerabilities": {
                    "injection": "When untrusted data is sent to an interpreter as part of a command or query",
                    "broken_authentication": "Incorrect implementation of authentication and session management",
                    "xss": "Cross-site scripting occurs when an app includes untrusted data in a web page"
                },
                "security_tools": {
                    "firewall": "Network security system that monitors and controls incoming/outgoing traffic",
                    "ids": "Intrusion Detection System that monitors network or systems for malicious activity",
                    "encryption": "Process of encoding information to prevent unauthorized access"
                }
            }
            
            # Save the basic knowledge
            with open(knowledge_file, 'w') as f:
                json.dump(basic_knowledge, f, indent=4)
                
            return basic_knowledge
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response to a cybersecurity-related query."""
        # Check if it's a vulnerability assessment request
        if any(kw in query.lower() for kw in ["vulnerability", "assessment", "scan", "secure"]):
            return self._provide_security_assessment(query, context)
        
        # Check if it's about a specific attack or threat
        if any(kw in query.lower() for kw in ["attack", "threat", "exploit", "hack"]):
            return self._explain_security_threat(query, context)
        
        # Check if it's about security best practices
        if any(kw in query.lower() for kw in ["best practice", "guideline", "recommendation"]):
            return self._provide_security_practices(query, context)
        
        # General security question
        return self._answer_security_question(query, context)
    
    def _provide_security_assessment(self, query: str, context: List[str]) -> str:
        """Provide a security assessment response."""
        response = "Based on your security assessment request, here are key considerations:\n\n"
        
        # Add common vulnerability checks
        response += "## Security Assessment Checklist\n\n"
        response += "1. **Input Validation**\n   - Ensure all user inputs are validated and sanitized\n   - Implement parameterized queries for database operations\n\n"
        response += "2. **Authentication & Authorization**\n   - Use multi-factor authentication where possible\n   - Implement proper access controls with least privilege\n\n"
        response += "3. **Data Protection**\n   - Encrypt sensitive data in transit and at rest\n   - Implement secure key management\n\n"
        
        response += "To conduct a thorough assessment, I would need more specific information about your system architecture, technologies used, and security requirements."
        
        return response
    
    def _explain_security_threat(self, query: str, context: List[str]) -> str:
        """Explain a security threat or attack."""
        # Identify which threat might be referenced
        threat_keywords = {
            "sql injection": ["sql", "injection", "database"],
            "xss": ["xss", "cross-site", "script"],
            "csrf": ["csrf", "cross-site", "request", "forgery"],
            "ddos": ["ddos", "denial", "service"],
            "ransomware": ["ransomware", "ransom", "encrypt"],
            "phishing": ["phish", "email", "social"]
        }
        
        # Find matching threats
        possible_threats = []
        for threat, keywords in threat_keywords.items():
            if any(kw in query.lower() for kw in keywords):
                possible_threats.append(threat)
        
        # Default to a general explanation if no specific threat is identified
        if not possible_threats:
            possible_threats = ["general security threats"]
        
        # Provide explanation
        threat = possible_threats[0]
        response = f"## Understanding {threat.upper()}\n\n"
        
        if threat == "sql injection":
            response += "SQL Injection is an attack where malicious SQL statements are inserted into entry fields for execution.\n\n"
            response += "### How it works:\n"
            response += "1. Attackers find vulnerable input fields that directly use user input in SQL queries\n"
            response += "2. They inject code like `' OR 1=1 --` to manipulate query logic\n"
            response += "3. This can lead to unauthorized data access, modification, or deletion\n\n"
            response += "### Prevention:\n"
            response += "- Use parameterized queries or prepared statements\n"
            response += "- Implement input validation and sanitization\n"
            response += "- Apply the principle of least privilege to database accounts\n"
        elif threat == "xss":
            response += "Cross-Site Scripting (XSS) allows attackers to inject client-side scripts into web pages viewed by other users.\n\n"
            response += "### How it works:\n"
            response += "1. Attacker finds a way to inject JavaScript into a webpage\n"
            response += "2. When other users view the page, the script executes in their browser\n"
            response += "3. This can steal cookies, session tokens, or redirect users to malicious sites\n\n"
            response += "### Prevention:\n"
            response += "- Sanitize and validate all user inputs\n"
            response += "- Implement Content-Security-Policy headers\n"
            response += "- Use context-aware output encoding\n"
        else:
            # General security threat information
            response += "Security threats typically exploit vulnerabilities in systems through various attack vectors.\n\n"
            response += "### Common Attack Patterns:\n"
            response += "1. Exploiting unpatched software vulnerabilities\n"
            response += "2. Taking advantage of misconfigurations\n"
            response += "3. Using social engineering to bypass technical controls\n\n"
            response += "### General Prevention Strategies:\n"
            response += "- Keep systems and software updated\n"
            response += "- Implement defense in depth\n"
            response += "- Train users on security awareness\n"
            response += "- Regularly audit and test security controls\n"
            
        return response
    
    def _provide_security_practices(self, query: str, context: List[
