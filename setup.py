from setuptools import setup, find_packages

setup(
    name="datahub-langchain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "acryl-datahub[datahub-rest]",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "openai",
        "langsmith",
        "python-dotenv",
    ],
)
