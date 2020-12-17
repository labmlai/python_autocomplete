import setuptools

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml_python_autocomplete',
    version='0.0.1',
    author="Varuna Jayasiri",
    author_email="vpjayasiri@gmail.com",
    description="A simple model that learns to predict Python source code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lab-ml/python_autocomplete",
    project_urls={
        'Documentation': 'https://lab-ml.com/'
    },
    packages=setuptools.find_packages(exclude=('test',
                                               'test.*')),
    install_requires=['labml>=0.4.74',
                      'labml_helpers>=0.4.70',
                      'labml_nn>=0.4.70'
                      'torch',
                      'einops',
                      'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning',
)
