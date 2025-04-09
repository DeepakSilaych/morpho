from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="morpho",
    version="0.1.0",
    author="Deepak Silaych",
    author_email="deepaksilaych@gmail.com",
    description="A modular, extensible, and diverse tokenization library for NLP and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepaksilaych/morpho",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
        "tqdm>=4.42.0",
    ],
    extras_require={
        "sentencepiece": ["sentencepiece>=0.1.91"],
        "torch": ["torch>=1.0.0"],
        "tensorflow": ["tensorflow>=2.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "morpho=cli.main:main",
        ],
    },
) 