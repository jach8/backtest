from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="btest8",
    version="0.1.0",
    author="Jerald Achaibar",
    description="A price data management system for backtesting trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jach8/backtest",
    packages=find_packages(include=["*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "typing",
        "datetime",
        "tqdm",
        "matplotlib"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.9",
            "pylint>=2.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "backtest-price-data=main:main",
        ],
    }
)