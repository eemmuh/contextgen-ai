from typing import List, Dict, Optional
from src.embedding.embedding_manager import EmbeddingManager

class RAGManager:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        num_examples: int = 3,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the RAG manager for handling retrieval and prompt augmentation.
        
        Args:
            embedding_manager: Instance of EmbeddingManager
            num_examples: Number of similar examples to retrieve
            similarity_threshold: Minimum similarity score for retrieved examples
        """
        self.embedding_manager = embedding_manager
        self.num_examples = num_examples
        self.similarity_threshold = similarity_threshold
    
    def retrieve_similar_examples(self, query: str) -> List[Dict]:
        """
        Retrieve similar examples based on the query.
        
        Args:
            query: User's input query
            
        Returns:
            List of similar examples with their metadata and similarity scores
        """
        # Compute query embedding
        query_embedding = self.embedding_manager.compute_text_embedding(query)
        
        # Search for similar examples
        results = self.embedding_manager.search(
            query_embedding,
            k=self.num_examples
        )
        
        # Filter results based on similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity_score'] >= self.similarity_threshold
        ]
        
        return filtered_results
    
    def augment_prompt(
        self,
        query: str,
        similar_examples: List[Dict]
    ) -> str:
        """
        Augment the original prompt with information from similar examples.
        
        Args:
            query: Original user query
            similar_examples: List of similar examples retrieved
            
        Returns:
            Augmented prompt string
        """
        if not similar_examples:
            return query
        
        # Extract relevant information from similar examples
        style_info = []
        tag_info = []
        
        for example in similar_examples:
            metadata = example['metadata']
            
            if metadata.get('style'):
                style_info.append(metadata['style'])
            
            if metadata.get('tags'):
                tag_info.extend(metadata['tags'])
        
        # Create augmented prompt
        augmented_parts = [query]
        
        if style_info:
            unique_styles = list(set(style_info))
            style_text = f"in the style of {', '.join(unique_styles)}"
            augmented_parts.append(style_text)
        
        if tag_info:
            unique_tags = list(set(tag_info))
            tag_text = f"with elements of {', '.join(unique_tags[:5])}"  # Limit to top 5 tags
            augmented_parts.append(tag_text)
        
        return ", ".join(augmented_parts)
    
    def process_query(self, query: str) -> Dict:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary containing:
            - original_query: Original user query
            - augmented_prompt: Augmented prompt for image generation
            - similar_examples: Retrieved similar examples
        """
        # Retrieve similar examples
        similar_examples = self.retrieve_similar_examples(query)
        
        # Augment the prompt
        augmented_prompt = self.augment_prompt(query, similar_examples)
        
        return {
            'original_query': query,
            'augmented_prompt': augmented_prompt,
            'similar_examples': similar_examples
        } 