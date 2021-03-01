from setuptools import setup, find_packages
import pathlib

VERSION = '0.0.7' 
BASE = pathlib.Path(__file__).parent
README = (BASE / "README.md").read_text()
AUTHORS = 'Gustavo Viera, Alfredo Reyes, Jorge Morgado, Ernesto Altushler'
DESCRIPTION = 'A package for tracking and analysing objects trajectories'
# Setting up
setup(
        name="yupitest", 
        version=VERSION,
        author=AUTHORS,
        author_email="areyes@fisica.uh.cu",
        description=DESCRIPTION,
        long_description=README,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=[],
        include_package_data=True,
        license='MIT',        
        keywords=['data science', 'tracking'],
        classifiers= [
            "Programming Language :: Python :: 3",
        ]
)