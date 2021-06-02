import setuptools

exec(open("./antisplodge/_version.py").read()) # this will provide __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="antisplodge", # Replace with your own username
    version=__version__,
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
    package_dir={"": "antisplodge"},
    packages=[''],
    python_requires=">=3.6",
    install_requires=[
    'numpy>=1.17.2',
    'pandas>=0.25.3',
    'scikit-learn>=0.22.1',
    'torch>=1.8.1'
    ]
)
