from setuptools import find_packages, setup


def _requires_from_file(filename):
    with open(filename, "r") as f:
        modules = f.readlines()
    return modules


setup(
    name="fbts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=_requires_from_file("requirements.txt"),
    include_package_data=True,
)
