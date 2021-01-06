from setuptools import setup, find_namespace_packages
setup(

    name="py-calibrate",
    version="0.0.3",
    author="Oliver Batchelor",
    author_email="saulzar@gmail.com",
    description="Python structs and tables using dot notation",
    url="https://github.com/saulzar/py-calibrate",
    packages=find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python>=4.2.0.32",
        "natsort",
        "cached-property",
        "py-structs"
    ]

    python_requires='>=3.7',
)
