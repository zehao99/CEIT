import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='CEIT',
    version='0.0.1',
    description='Python Package for EIT(Electric Impedance Tomography)-like problems using Gauss-Newton method.',
    url="https://github.com/zehao99/CEIT",
    author="Li Zehao",
    install_requires=["numpy", "matplotlib", "progressbar2"],
    author_email="zehaoli99@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=['docs', 'test', 'test_data']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)