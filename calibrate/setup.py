from setuptools import setup, find_namespace_packages
setup(

    name="py-structs",
    version="0.2.0",
    author="Oliver Batchelor",
    author_email="saulzar@gmail.com",
    description="Python structs and tables using dot notation",
    url="https://github.com/saulzar/structs",
    packages=find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
