from setuptools import setup, find_packages

setup(
    name="popgen",
    version="3.0.1",
    author="Fan Yu",
    author_email="fanyu4@asu.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "popgen=popgen.cli:main"
        ]
    },
)





setup(
    name="popgen",
    version="1.0.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A synthetic population generation tool.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/popgen",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
