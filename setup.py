import os
from setuptools import setup, find_packages


def read(fname):
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


reqs = ["autograd>=1.2", "cupy>=4.0"]


setup(
    name="autograd-cupy",
    version="0.1",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    description=("autograd wrapper for CuPy"),
    license="MIT",
    keywords="machine learning, deep learning",
    url="http://github.com/ericmjl/autograd-cupy",
    packages=find_packages(),
    package_data={"": ["README.md", "LICENSE"]},
    install_requires=reqs,
    long_description=read("README.md"),
)
