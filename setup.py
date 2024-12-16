from setuptools import setup, find_packages

setup(
    name="smartML",
    version="0.1.4",
    description="An ML library for various tasks!",
    author="Şuayp Talha Kocabay",
    author_email="kocabaysuayptalha08@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
