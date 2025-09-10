from setuptools import setup, find_packages

setup(
    name="k-diffusion-fork",
    version="0.0.1",
    packages=find_packages(include=["k_diffusion", "k_diffusion.*"]),
)