"""
Setup script for FRB-RFI Detector
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="frb-rfi-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="SAM-based FRB detection in the presence of RFI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arpan-52/frb-rfi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "h5py>=3.6.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
