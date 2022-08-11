import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_camera",
    version="0.1.0",
    author="James Cotton",
    author_email="peabody124@gmail.com",
    description="Library for multi-camera acquisition and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peabody124/MultiCameraTracking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['anipose', 'opencv-python', 'numpy', 'tqdm']
)