import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AntiSplodge", # Replace with your own username
    version="0.0.1",
    author="Jesper Beltoft Lund",
    author_email="Jesper.Lund@hpi.de",
    description="AntiSplodge: A neural network-based spatial transcriptomics deconvolution pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HealthML/AntiSplodge",
    project_urls={
        "Bug Tracker": "https://github.com/HealthML/AntiSplodge/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy>=1.17.2', 'pandas>=0.25.3'
                     ]
)
