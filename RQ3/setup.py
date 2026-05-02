from setuptools import setup, find_packages

setup(

    name="craft-sanitization",

    version="1.0.0",

    description="CRAFT: Context-Routing Artifact-Faithful Transformations for Privacy-Preserving LLM Prompt Sanitization",

    packages=find_packages(),

    python_requires=">=3.10",

    install_requires=[

        "presidio-analyzer>=2.2",

        "presidio-anonymizer>=2.2",

        "spacy>=3.7",

        "requests>=2.31",

        "scipy>=1.11",

    ],

)
