"""
FMCaps: Integrating Foundation Models with Capsule Networks for Enhanced WSSS
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fmcaps",
    version="1.0.0",
    author="FMCaps Authors",
    author_email="",
    description="Integrating Foundation Models with Capsule Networks for Enhanced Weakly-Supervised Semantic Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FMCaps",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "sam": ["segment-anything"],
        "gdino": ["groundingdino-py"],
        "full": ["segment-anything", "groundingdino-py"],
    },
    entry_points={
        "console_scripts": [
            "fmcaps-train-voc=train.voc_train_capsule:main",
            "fmcaps-train-coco=train.coco_train_capsule:main",
            "fmcaps-eval=tools.evaluate:main",
            "fmcaps-infer=tools.inference:main",
            "fmcaps-pseudo=tools.generate_pseudo_labels:main",
        ],
    },
)

