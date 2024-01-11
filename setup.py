# coding=utf-8
# Copyright 2022 Hugging Face Inc
#
# Lint as: python3
# pylint: enable=line-too-long
"""Hugging Face Competitions
"""
import os

from setuptools import find_packages, setup


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

QUALITY_REQUIRE = [
    "black~=23.0",
    "isort==5.13.2",
    "flake8==7.0.0",
    "mypy==1.8.0",
]

TEST_REQUIRE = ["pytest", "pytest-cov"]

EXTRAS_REQUIRE = {
    "dev": QUALITY_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "test": QUALITY_REQUIRE + TEST_REQUIRE,
    "docs": QUALITY_REQUIRE + TEST_REQUIRE + ["hf-doc-builder"],
}

with open("requirements.txt", encoding="utf-8") as f:
    INSTALL_REQUIRES = f.read().splitlines()

setup(
    name="competitions",
    description="Hugging Face Competitions",
    long_description=LONG_DESCRIPTION,
    author="HuggingFace Inc.",
    url="https://github.com/huggingface/competitions",
    download_url="https://github.com/huggingface/competitions/tags",
    packages=find_packages("."),
    entry_points={"console_scripts": ["competitions=competitions.cli.competitions:main"]},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.10",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="huggingface competitions machine learning ai nlp tabular",
    data_files=[
        (
            "templates",
            [
                "competitions/templates/index.html",
            ],
        ),
    ],
    include_package_data=True,
)
