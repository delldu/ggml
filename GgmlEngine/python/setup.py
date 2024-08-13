"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 30 Jan 2024 11:51:01 PM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="ggml_engine",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="Ggml Engine Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/GgmlEngine.git",
    packages=["ggml_engine"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 2.0.0",
        "numpy >= 1.21.3",
        "safetensors >= 0.3.1",
        "gguf >= 0.6.0",
    ],
)
