from setuptools import setup, find_namespace_packages
setup(

    name="multical",
    version="0.1.0",
    author="Oliver Batchelor",
    author_email="saulzar@gmail.com",
    description="Flexible multi-camera multi-board camera calibration library and application.",
    url="https://github.com/saulzar/multical",
    packages=find_namespace_packages(),
    scripts=['multical/app/multical'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera"
    ],

    include_package_data=True,

    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python>=4.2.0.0",
        "opencv-contrib-python",

        "natsort",
        "cached-property",
        "py-structs>=0.2.1"
    ],

    extras_require={
        'interactive': ['matplotlib', 'qtpy', 'pyvista-qt', 'pyvista'],
    },

    python_requires='>=3.6',
)
