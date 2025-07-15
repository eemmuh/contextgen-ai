import torch
from diffusers import StableDiffusionPipeline
from typing import Dict, Optional, List
import os
from PIL import Image

class ImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_inference_steps: int = 20,  # Reduced from 50 to 20 for speed
        guidance_scale: float = 7.5
    ):
        """
        Initialize the image generator.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run the model on
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Initialize the pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipeline = self.pipeline.to(device)
        
        if device == "cuda":
            self.pipeline.enable_attention_slicing()
            # Additional optimizations for speed
            self.pipeline.enable_model_cpu_offload()  # Offload to CPU when not in use
            self.pipeline.enable_vae_slicing()  # Reduce memory usage
            self.pipeline.enable_sequential_cpu_offload()  # Sequential offloading
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        fast_mode: bool = False
    ) -> List[Image.Image]:
        """
        Generate images based on the prompt.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to guide generation away from certain elements
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Adjust parameters for fast mode
        inference_steps = 10 if fast_mode else self.num_inference_steps
        guidance_scale = 5.0 if fast_mode else self.guidance_scale
        
        # Generate images
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images
        
        return images
    
    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "generated"
    ) -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images: List of PIL Images to save
            output_dir: Directory to save images in
            prefix: Prefix for image filenames
            
        Returns:
            List of paths to saved images
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{i+1}.png"
            path = os.path.join(output_dir, filename)
            image.save(path)
            saved_paths.append(path)
        
        return saved_paths
    
    def generate_from_rag_output(
        self,
        rag_output: Dict,
        output_dir: str,
        num_images: int = 1,
        seed: Optional[int] = None,
        fast_mode: bool = False
    ) -> Dict:
        """
        Generate images using the output from the RAG process.
        
        Args:
            rag_output: Dictionary containing RAG process output
            output_dir: Directory to save generated images
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - original_prompt: Original user query
            - augmented_prompt: Augmented prompt used for generation
            - generated_images: List of paths to generated images
            - similar_examples: Retrieved similar examples
        """
        # Generate images using the augmented prompt
        images = self.generate_image(
            prompt=rag_output['augmented_prompt'],
            num_images=num_images,
            seed=seed,
            fast_mode=fast_mode
        )
        
        # Save generated images
        saved_paths = self.save_images(
            images=images,
            output_dir=output_dir,
            prefix=f"rag_{seed if seed is not None else 'random'}"
        )
        
        return {
            'original_prompt': rag_output['original_query'],
            'augmented_prompt': rag_output['augmented_prompt'],
            'generated_images': saved_paths,
            'similar_examples': rag_output['similar_examples']
        } 