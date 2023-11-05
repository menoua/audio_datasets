import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audio_datasets",
    version="0.0.1",
    author="Menoua Keshishian",
    author_email="menoua.k@columbia.edu",
    description="Wrapper for loading and batching audio files from well-known datasets for model training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/menoua/audio_datasets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
