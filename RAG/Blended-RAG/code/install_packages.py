import os

# List of packages to install
packages = [
    "requests",
    "pandas",
    "numpy",
    "pillow",
    "nltk",
    "rouge-score",
    "beautifulsoup4",
    "torch",
    "py-simhash",
    "bleurt",
    "matplotlib",
    "tensorflow",
    "elasticsearch"
]

# Install each package using pip
for package in packages:
    os.system(f"pip install {package}")
