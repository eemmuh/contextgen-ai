from setuptools import setup, find_packages

setup(
    name="image-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "Pillow",
        "tqdm",
        "pycocotools",
    ],
    python_requires=">=3.8",
) 


