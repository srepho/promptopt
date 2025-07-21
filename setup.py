from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="promptopt",
    version="0.1.0",
    author="promptopt team",
    author_email="team@promptopt.dev",
    description="Enterprise prompt optimization framework combining DSPy and GRPO approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/promptopt/promptopt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)