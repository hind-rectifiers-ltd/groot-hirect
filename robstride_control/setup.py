#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tianrking Control Python Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="robstride-control",
    version="1.0.0",
    author="tianrking",
    author_email="",
    description="Python implementation of tianrking motor control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tianrking/robstride-control",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "robstride-position=src.position_control:main",
            "robstride-speed=src.speed_control:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)