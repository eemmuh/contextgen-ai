import os
import requests
import time
from PIL import Image
from io import BytesIO
import pandas as pd


class PexelsDatasetDownloader:
    def __init__(self, access_key: str):
        """
        Initialize the Pexels dataset downloader.

        Args:
            access_key: Pexels API access key
        """
        self.access_key = access_key
        self.base_url = "https://api.pexels.com/v1"
        self.headers = {"Authorization": access_key}

        # Define categories and their search terms
        self.categories = {
            "Landscapes": ["mountain landscape", "ocean view", "forest scene"],
            "Industrial": [
                "industrial architecture",
                "urban decay",
                "factory interior",
            ],
            "Floral": ["flower arrangement", "botanical art", "floral pattern"],
            "Black & White": ["black and white photography", "monochrome art"],
            "Minimalist": ["minimalist art", "minimal design", "simple composition"],
            "Surrealism": ["surreal art", "dreamlike scene", "fantasy art"],
            "Seasonal": ["seasonal landscape", "seasonal decoration"],
        }

    def search_photos(self, query: str, per_page: int = 10):
        """Search for photos on Pexels."""
        endpoint = f"{self.base_url}/search"
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape",  # Most wall art is landscape orientation
        }

        response = requests.get(endpoint, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()["photos"]
        else:
            print(f"Error searching photos: {response.status_code}")
            return []

    def download_image(self, url: str, save_path: str) -> bool:
        """Download and save an image."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(save_path)
                return True
        except Exception as e:
            print(f"Error downloading image: {e}")
        return False

    def create_dataset(self, output_dir: str, images_per_category: int = 5):
        """Create the dataset by downloading images for each category."""
        os.makedirs(output_dir, exist_ok=True)

        metadata = []
        image_count = 0

        for category, search_terms in self.categories.items():
            print(f"\nProcessing category: {category}")

            for term in search_terms:
                photos = self.search_photos(term, per_page=images_per_category)

                for photo in photos:
                    image_count += 1
                    filename = f"image_{image_count:03d}.jpg"
                    save_path = os.path.join(output_dir, filename)

                    if self.download_image(photo["src"]["large"], save_path):
                        # Create metadata entry
                        metadata.append(
                            {
                                "image_filename": filename,
                                "description": photo.get("alt", term),
                                "tags": ",".join(photo.get("tags", [])[:5]),
                                "style": category,
                                "category": category,
                            }
                        )

                    # Respect Pexels API rate limits
                    time.sleep(0.1)

        # Save metadata
        df = pd.DataFrame(metadata)
        df.to_csv(
            os.path.join(os.path.dirname(output_dir), "metadata.csv"), index=False
        )
        print(f"\nDataset created successfully with {len(metadata)} images!")


def main():
    # You'll need to replace this with your Pexels API access key
    access_key = os.getenv("PEXELS_API_KEY")
    if not access_key:
        print("Please set your Pexels API key in the .env file")
        return

    downloader = PexelsDatasetDownloader(access_key)
    downloader.create_dataset("data/images")


if __name__ == "__main__":
    main()
