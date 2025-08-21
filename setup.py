#!/usr/bin/env python3
"""
Setup script for Prism: Wideband RF Neural Radiance Fields
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prism-rf",
    version="0.1.0",
    author="Prism Project Team",
    author_email="contact@prism-project.org",
    description="Wideband RF Neural Radiance Fields for OFDM Communication",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tagsysx/Prism",
    project_urls={
        "Bug Reports": "https://github.com/tagsysx/Prism/issues",
        "Source": "https://github.com/tagsysx/Prism",
        "Documentation": "https://github.com/tagsysx/Prism#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prism-train=prism.scripts.train:main",
            "prism-test=prism.scripts.test:main",
            "prism-demo=prism.scripts.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "neural-radiance-fields",
        "rf-signals",
        "ofdm",
        "mimo",
        "wireless-communication",
        "deep-learning",
        "pytorch",
        "computer-vision",
        "signal-processing",
    ],
)
