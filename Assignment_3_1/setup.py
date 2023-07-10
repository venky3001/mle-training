from setuptools import setup

setup(
    name="custom_package",
    version="0.3",
    description="Median housing value prediction",
    author="TA-Venkatesh",
    author_email="venkatesh.talasi@tigeranalytics.com",
    packages=["src"],
    install_requires=["numpy", "pandas", "sklearn", "scipy", "six", "argparse"],
)
