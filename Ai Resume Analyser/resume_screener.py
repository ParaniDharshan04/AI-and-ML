import re
from typing import Optional

class ResumeMatcherRAG:
    """
    AI-powered Resume Matcher using Retrieval-Augmented Generation (RAG).
    Loads embedding and LLM models, preprocesses text, and sets up a vector store for semantic retrieval.
    """
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', llm_model_name: str = 'google/flan-t5-small'):
        """
        Initialize the ResumeMatcherRAG with specified embedding and LLM models.
        Loads models and prepares a placeholder for the vector store.
        Args:
            embedding_model_name (str): Name of the sentence transformer model for embeddings.
  b          llm_model_name (str): Name of the LLM for RAG generation/re-ranking.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Required packages not found. Please install 'sentence-transformers' and 'transformers'.") from e

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{embedding_model_name}': {e}")

        # Load LLM model and tokenizer
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model '{llm_model_name}': {e}")

        # Placeholder for vector store (to be implemented later)
        self.vector_store = None
        self.resume_chunks = None

    def _preprocess_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Basic text cleaning: lowercasing, removing extra whitespace, and optional stopword removal.
        Args:
            text (str): Input text to clean.
            remove_stopwords (bool): Whether to remove common English stopwords (default: False).
        Returns:
            str: Cleaned text.
        Design Note:
            Stopword removal is optional because some job requirements may use words that are stopwords (e.g., 'and', 'or').
        """
        # Lowercase and remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip().lower())
        if remove_stopwords:
            try:
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                tokens = cleaned.split()
                cleaned = ' '.join([t for t in tokens if t not in ENGLISH_STOP_WORDS])
            except ImportError:
                # If sklearn is not available, skip stopword removal
                pass
        return cleaned 

    def _chunk_text(self, text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
        
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - chunk_overlap
        return chunks

    def _generate_embeddings(self, texts: list[str]):
        
        if not texts:
            raise ValueError("Input text list for embedding is empty.")
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required for embedding output.") from e
        # The embedding model returns a list or np.ndarray
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings) 

    def _build_vector_store(self, documents: list[str]):
       
        if not documents:
            raise ValueError("No documents provided to build the vector store.")
        embeddings = self._generate_embeddings(documents)
        # Store for later retrieval
        self.vector_store = embeddings
        self.resume_chunks = documents
        return embeddings, documents 

    def _retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> list[str]:
        
        if self.vector_store is None or self.resume_chunks is None:
            raise ValueError("Vector store or resume chunks not initialized. Call _build_vector_store first.")
        if not query:
            return []
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required for similarity computation.") from e
        
        query_clean = self._preprocess_text(query)
        query_emb = self._generate_embeddings([query_clean])[0] 
        
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sims = np.dot(self.vector_store, query_emb) / (
            np.linalg.norm(self.vector_store, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.resume_chunks[i] for i in top_indices] 

    def _augment_prompt_with_retrieval(self, job_description_chunk: str, retrieved_resume_chunks: list[str]) -> str:
        
        prompt = (
            f"Job Requirement: {job_description_chunk}\n"
            f"Relevant Resume Sections:\n"
        )
        for i, chunk in enumerate(retrieved_resume_chunks, 1):
            prompt += f"[{i}] {chunk}\n"
        prompt += (
            "\nBased on the above, does the candidate meet this requirement? "
            "Reply ONLY with 'Yes' if the candidate's resume explicitly mentions this requirement, and quote the relevant line from the resume as evidence. "
            "If not, reply 'No' and explain why not. Do not reply 'Possibly'. "
            "If you reply 'Yes', you MUST include the exact line from the resume that matches. "
            "If you reply 'No', explain what is missing. "
        )
        return prompt

    def _get_llm_response(self, prompt: str) -> str:
        """
        Sends the augmented prompt to the loaded LLM and returns its response.
        Args:
            prompt (str): The prompt to send to the LLM.
        Returns:
            str: LLM's response.
        """
        # Tokenize and generate response
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.llm_model.generate(**inputs, max_new_tokens=128)
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def match_resume(self, job_description: str, resume_text: str) -> dict:
        """
        Orchestrates the RAG flow: preprocesses, chunks, builds vector store, retrieves, prompts LLM, and aggregates results.
        Args:
            job_description (str): The full job description text.
            resume_text (str): The full resume text.
        Returns:
            dict: {
                'relevance_score': float (0-1),
                'matched_skills': list[str],
                'highlighted_sections': list[str],
                'llm_assessments': list[dict] (per JD chunk)
            }
        """
        # Debug prints to see what the model is analyzing
        print("\n=== DEBUG: Job Description ===\n", job_description)
        print("\n=== DEBUG: Resume Text ===\n", resume_text)
        # Preprocess and chunk
        jd_clean = self._preprocess_text(job_description)
        resume_clean = self._preprocess_text(resume_text)
        # Use smaller chunk sizes and more overlap for better matching
        jd_chunks = self._chunk_text(jd_clean, chunk_size=20, chunk_overlap=10)  # smaller chunks for requirements
        resume_chunks = self._chunk_text(resume_clean, chunk_size=60, chunk_overlap=20)
        # Build vector store for resume
        self._build_vector_store(resume_chunks)
        llm_assessments = []
        matched_skills = set()
        highlighted_sections = set()
        match_count = 0
        for jd_chunk in jd_chunks:
            relevant_resume_chunks = self._retrieve_relevant_chunks(jd_chunk, top_k=3)
            # Make the prompt more lenient and instructive
            prompt = (
                f"Job Requirement: {jd_chunk}\n"
                f"Relevant Resume Sections:\n"
            )
            for i, chunk in enumerate(relevant_resume_chunks, 1):
                prompt += f"[{i}] {chunk}\n"
            prompt += (
                "\nBased on the above, does the candidate meet this requirement? "
                "Reply ONLY with 'Yes' if the candidate's resume explicitly mentions this requirement, and quote the relevant line from the resume as evidence. "
                "If not, reply 'No' and explain why not. Do not reply 'Possibly'. "
                "If you reply 'Yes', you MUST include the exact line from the resume that matches. "
                "If you reply 'No', explain what is missing. "
            )
            llm_response = self._get_llm_response(prompt)
            # Debug: print LLM response and matched chunks
            print(f"\n=== DEBUG: JD Chunk ===\n{jd_chunk}")
            print(f"=== DEBUG: Relevant Resume Chunks ===\n{relevant_resume_chunks}")
            print(f"=== DEBUG: LLM Response ===\n{llm_response}")
            # Stricter matching: only count 'Yes' as a match
            assessment = {
                'jd_chunk': jd_chunk,
                'resume_chunks': relevant_resume_chunks,
                'llm_response': llm_response
            }
            if llm_response.lower().startswith('yes'):
                match_count += 1
                highlighted_sections.update(relevant_resume_chunks)
                # Try to extract skills/experiences from the response
                import re
                skills = re.findall(r"skills?[:\-\s]+([\w, ]+)", llm_response, re.IGNORECASE)
                for skill_str in skills:
                    for skill in skill_str.split(','):
                        skill = skill.strip()
                        if skill:
                            matched_skills.add(skill)
            llm_assessments.append(assessment)
        relevance_score = match_count / len(jd_chunks) if jd_chunks else 0.0
        return {
            'relevance_score': round(relevance_score, 2),
            'matched_skills': list(matched_skills),
            'highlighted_sections': list(highlighted_sections),
            'llm_assessments': llm_assessments
        }

if __name__ == "__main__":
    # Example usage with dummy data
    jd = """
    We are seeking a Python developer with experience in machine learning, data analysis, and cloud deployment. Must be proficient in Python, familiar with scikit-learn, and have deployed models to AWS or Azure.
    """
    resume = """
    John Doe is a software engineer skilled in Python and Java. He has built machine learning models using scikit-learn and pandas, and deployed solutions on AWS. He also has experience with REST APIs and Docker.
    """
    screener = ResumeMatcherRAG()
    result = screener.match_resume(jd, resume)
    print("Relevance Score:", result['relevance_score'])
    print("Matched Skills:", result['matched_skills'])
    print("Highlighted Sections:")
    for section in result['highlighted_sections']:
        print("-", section)
    print("\nLLM Assessments:")
    for assess in result['llm_assessments']:
        print(f"JD: {assess['jd_chunk']}")
        print(f"LLM: {assess['llm_response']}")
        print() 