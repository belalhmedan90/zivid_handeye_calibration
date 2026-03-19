#!/usr/bin/python3

from setuptools import setup, find_packages

PACKAGE_NAME = "zivid_handeye_calibration"

with open('requirements.txt', 'r', encoding='utf-8') as f:
    install_deps = [line.strip() for line in f.readlines() if line.strip()]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    version="0.0.0",
    author="Belal",
    author_email="s.com",
    description=("ZIVID HandEye calibration package."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=install_deps,
    python_requires=">=3.8",
    # Copies dist/pytransform dir into production
    package_data={},
)
