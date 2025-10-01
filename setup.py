"""
Setup script for the Tocantins Index calculator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="impact-score-calculator",
    version="1.0.0",
    author="Isaque Carvalho Borges",
    author_email="isaque@ecoacaobrasil.org",
    description="Package to calculate the Tocantins Index for compounded quantification of intra-urban heat anomalies in Landsat imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EcoAcao-Brasil/Tocantins-Index/src/impact-score",
    project_urls={
        "Bug Reports": ".....", # Yet to be changed by the actual URL
        "Source": ".....",  # Yet to be changed by the actual URL
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "rasterio>=1.2.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
)
