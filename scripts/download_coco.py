import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.coco_downloader import COCODatasetDownloader


def main():
    print("Starting COCO dataset download and processing...")
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

    print("COCO dataset download and processing completed!")


if __name__ == "__main__":
    main()
