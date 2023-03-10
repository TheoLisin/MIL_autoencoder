import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MIL test assignment",
    author="Theo",
    author_email="theo.lisin@gmail.com",
    description="Python project",
    keywords="Python, CIFAR10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheoLisin/MIL_autoencoder",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    version="0.1.0",
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 1 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tqdm",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm",
        "numpy",
        "marshmallow_dataclass",
        "matplotlib",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "jupyterlab",
            "torchsummary",
            "ipywidgets",
            "wemake-python-styleguide",
            "mypy",
            "black",
        ],
        "tests": [
            "pytest",
            "pytest-dotenv",
        ],
    },
    entry_points={
        "console_scripts": [
            "train = mil_autoencoder.__main__:main",
        ],
    },
)
