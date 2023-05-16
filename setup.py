from setuptools import setup, find_namespace_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(

    name="multical",
    version="0.2.1",
    author="Oliver Batchelor",
    author_email="saulzar@gmail.com",
    description="Flexible multi-camera multi-board camera calibration library and application.",
    url="https://github.com/saulzar/multical",
    packages=find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera"
    ],

    entry_points={
        'console_scripts': [
            'multical = multical.app.multical:cli',
        ],
    },

    long_description=long_description,
    long_description_content_type='text/markdown',

    include_package_data=True,

    install_requires = [
        "numpy",
        "numba",
        "scipy",
        "matplotlib",
        "opencv-contrib-python>=4.5.0.0",
        "simple-parsing",
        "natsort",
        "cached-property",
        "py-structs<1.0",
        "palettable",
        "numpy-quaternion",
        "omegaconf",
        "packaging",
        "tqdm"
    ],
    
    extras_require={
        'interactive': ['matplotlib', 'qtpy', 'pyvistaqt', 'pyvista', 'colour', 'qtawesome'],
    },

    python_requires='>=3.6',
)
