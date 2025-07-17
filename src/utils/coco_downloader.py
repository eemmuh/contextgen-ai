import os
import requests
import json
import zipfile
from pycocotools.coco import COCO
import urllib.request
from tqdm import tqdm


class COCODatasetDownloader:
    def __init__(self, data_dir: str = "data/coco"):
        """
        Initialize the COCO dataset downloader.

        Args:
            data_dir: Directory to download and store COCO dataset
        """
        self.data_dir = data_dir
        self.urls = {
            "train2017_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        }

        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "train2017"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "val2017"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "annotations"), exist_ok=True)

    def download_file(self, url: str, filename: str) -> bool:
        """Download a file with progress bar."""
        try:
            print(f"Downloading {filename}...")

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, "wb") as file:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=filename
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False

    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """Extract a zip file."""
        try:
            print(f"Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            os.remove(zip_path)  # Clean up zip file
            return True
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False

    def download_subset(self, split: str = "val2017", max_images: int = 1000):
        """
        Download a smaller subset of COCO for testing/development.

        Args:
            split: Dataset split to download (train2017 or val2017)
            max_images: Maximum number of images to download
        """
        print(f"Downloading COCO {split} subset ({max_images} images)...")

        # Download annotations first
        if not self.download_annotations():
            return False

        # Load COCO annotations
        ann_file = os.path.join(self.data_dir, "annotations", f"instances_{split}.json")
        coco = COCO(ann_file)

        # Get image IDs
        img_ids = list(coco.imgs.keys())[:max_images]

        # Download images
        images_dir = os.path.join(self.data_dir, split)
        os.makedirs(images_dir, exist_ok=True)

        successful_downloads = 0
        for img_id in tqdm(img_ids, desc=f"Downloading {split} images"):
            img_info = coco.imgs[img_id]
            img_url = img_info["coco_url"]
            img_filename = img_info["file_name"]
            img_path = os.path.join(images_dir, img_filename)

            if os.path.exists(img_path):
                successful_downloads += 1
                continue

            try:
                urllib.request.urlretrieve(img_url, img_path)
                successful_downloads += 1
            except Exception as e:
                print(f"Failed to download {img_filename}: {e}")

        print(f"Successfully downloaded {successful_downloads}/{len(img_ids)} images")
        return True

    def download_annotations(self) -> bool:
        """Download and extract COCO annotations."""
        annotations_zip = "annotations_trainval2017.zip"
        annotations_path = os.path.join(self.data_dir, annotations_zip)

        # Check if annotations already exist
        if os.path.exists(
            os.path.join(self.data_dir, "annotations", "instances_train2017.json")
        ):
            print("Annotations already exist, skipping download.")
            return True

        # Download annotations
        if not self.download_file(self.urls["annotations"], annotations_zip):
            return False

        # Extract annotations
        return self.extract_zip(annotations_path, self.data_dir)

    def download_full_dataset(self):
        """Download the complete COCO dataset (Warning: This is ~25GB)."""
        print("WARNING: This will download ~25GB of data. This may take several hours.")
        response = input("Do you want to continue? (y/N): ")

        if response.lower() != "y":
            print("Download cancelled.")
            return False

        # Download annotations
        if not self.download_annotations():
            return False

        # Download training images
        train_zip = "train2017.zip"
        if not self.download_file(self.urls["train2017_images"], train_zip):
            return False
        if not self.extract_zip(os.path.join(self.data_dir, train_zip), self.data_dir):
            return False

        # Download validation images
        val_zip = "val2017.zip"
        if not self.download_file(self.urls["val2017_images"], val_zip):
            return False
        if not self.extract_zip(os.path.join(self.data_dir, val_zip), self.data_dir):
            return False

        print("COCO dataset download completed!")
        return True

    def process_annotations(self, split: str = "val2017", max_images: int = None):
        """
        Process COCO annotations to create a simplified format for the RAG system.

        Args:
            split: Dataset split to process
            max_images: Maximum number of images to process (None for all)
        """
        print(f"Processing COCO annotations for {split}...")

        ann_file = os.path.join(self.data_dir, "annotations", f"instances_{split}.json")
        if not os.path.exists(ann_file):
            print(f"Annotation file not found: {ann_file}")
            return False

        coco = COCO(ann_file)

        # Get categories
        categories = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

        processed_annotations = []
        img_ids = list(coco.imgs.keys())

        if max_images:
            img_ids = img_ids[:max_images]

        for img_id in tqdm(img_ids, desc="Processing annotations"):
            img_info = coco.imgs[img_id]
            img_path = os.path.join(self.data_dir, split, img_info["file_name"])

            # Check if image file exists
            if not os.path.exists(img_path):
                continue

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Extract object categories
            object_names = [categories[ann["category_id"]] for ann in anns]
            unique_objects = list(set(object_names))

            processed_annotations.append(
                {
                    "id": img_id,
                    "image_path": img_path,
                    "file_name": img_info["file_name"],
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "captions": unique_objects,  # Using object names as captions
                    "num_objects": len(anns),
                }
            )

        # Save processed annotations
        output_file = os.path.join(self.data_dir, "processed_annotations.json")
        with open(output_file, "w") as f:
            json.dump(processed_annotations, f, indent=2)

        print(f"Processed annotations for {len(processed_annotations)} images")
        print(f"Saved to: {output_file}")
        return True


def main():
    """Download COCO dataset subset for development."""
    downloader = COCODatasetDownloader()

    print("COCO Dataset Downloader")
    print("1. Download subset (1000 images) - Recommended for development")
    print("2. Download full dataset (~25GB) - For production use")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        # Download a subset for development
        if downloader.download_subset(split="val2017", max_images=1000):
            downloader.process_annotations(split="val2017", max_images=1000)
    elif choice == "2":
        # Download full dataset
        if downloader.download_full_dataset():
            downloader.process_annotations(split="train2017")
            downloader.process_annotations(split="val2017")
    else:
        print("Invalid choice. Please run again and select 1 or 2.")


if __name__ == "__main__":
    main()
