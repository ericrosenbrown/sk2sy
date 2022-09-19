#!/usr/bin/env python3
from setuptools import setup
import os

# TODO make requirements work
repo_root = os.path.abspath(os.path.dirname(__file__))
with open(f"{repo_root}/requirements.txt", "r") as f:
    requirements = [x for x in f.read().splitlines() if x != ""]

setup(
    name="sk2sy",
    version="0.0.1",
    description="Skills to symbols.",
    url="https://github.com/ericrosenbrown/sk2sy",
    author="Eric Rosen",
    author_email="eric_rosen@brown.edu",
    packages=["sk2sy"],
    requires=requirements
)