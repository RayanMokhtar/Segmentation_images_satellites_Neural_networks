from setuptools import setup, find_packages

setup(
    name='flood-detection-cnn',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for flood detection using advanced CNN architectures and NDFI integration.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.18.0',
        'opencv-python>=4.5.0',
        'rasterio>=1.1.0',
        'matplotlib>=3.2.0',
        'scikit-learn>=0.24.0',
        'PyYAML>=5.3.0',
        'tqdm>=4.50.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)