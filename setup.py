from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath: str) -> List[str]:
    '''
    This function returns the list of requirements.
    '''
    HYPHEN_E_DOT = '-e .'
    requirements = []
    with open(filepath, 'r', encoding='utf-8') as file:  # Ensure file is read with correct encoding
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]  # Strip newline and empty lines

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Abdul Latif',
    author_email='labdul749@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
                                                                                                        