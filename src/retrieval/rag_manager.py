from typing import List, Dict, Optional
from src.embedding.embedding_manager import EmbeddingManager

class RAGManager:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        num_examples: int = 3,
        similarity_threshold: float = 0.1  # Lower threshold to catch more examples
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
        object_info = []
        scene_info = []
        
        for example in similar_examples:
            metadata = example['metadata']
            
            # Extract object names from COCO captions
            if metadata.get('tags'):
                object_info.extend(metadata['tags'])
            
            # Extract scene/context information from description
            if metadata.get('description'):
                description = metadata['description']
                # Look for scene indicators
                scene_keywords = ['indoor', 'outdoor', 'kitchen', 'living room', 'park', 'street', 'garden']
                for keyword in scene_keywords:
                    if keyword in description.lower():
                        scene_info.append(keyword)
        
        # Create augmented prompt
        augmented_parts = [query]
        
        # Add scene context
        if scene_info:
            unique_scenes = list(set(scene_info))
            scene_text = f"in {', '.join(unique_scenes[:2])} setting"  # Limit to 2 scenes
            augmented_parts.append(scene_text)
        
        # Add relevant objects that aren't already in the query
        if object_info:
            unique_objects = list(set(object_info))
            # Filter out objects already mentioned in the query
            query_lower = query.lower()
            new_objects = [obj for obj in unique_objects[:3] if obj.lower() not in query_lower]
            
            if new_objects:
                object_text = f"with {', '.join(new_objects)}"
                augmented_parts.append(object_text)
        
        # If no enhancement was added, add a generic style enhancement
        if len(augmented_parts) == 1:
            augmented_parts.append("in realistic style")
        
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
        
        # Show RAG enhancement summary
        if similar_examples:
            print(f"🎯 RAG Enhancement: Found {len(similar_examples)} similar examples")
            print(f"   Enhanced prompt: {augmented_prompt}")
        else:
            print("ℹ️  No similar examples found - using original prompt")
        
        return {
            'original_query': query,
            'augmented_prompt': augmented_prompt,
            'similar_examples': similar_examples
        } 