from setuptools import setup, find_packages

setup(
    name="popgen",
    version="3.0.1",
    author="Fan Yu",
    author_email="fanyu4@asu.edu",
    description="A population synthesis and sample weighting tool for transportation planning.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chnfanyu/PopGen3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.4.0",
        "pyyaml>=5.3"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",  # 如果是正式版，可改为 "5 - Production/Stable"
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "popgen=popgen.cli:main"
        ]
    },
    include_package_data=True,  # 确保附带非 Python 文件（如 YAML）
    zip_safe=False,  # 让 pip install 后仍然能修改包内文件
)
