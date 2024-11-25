from setuptools import setup, find_packages

setup(
    name="llm-lineage",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "acryl-datahub>=0.12.1.2",
        "acryl-datahub[datahub-rest]>=0.12.1.2",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "openai",
        "langsmith",
        "python-dotenv",
    ],
)
