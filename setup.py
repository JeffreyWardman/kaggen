from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='kaggen',
    version='0.0.0',
    description='A flexible deep learning pipeline library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jeffrey Wardman',
    author_email='jeffrey.wardman@yahoo.com.au',
    maintainer='Jeffrey Wardman',
    maintainer_email='jeffrey.wardman@yahoo.com.au',
    url='https://github.com/jeffreywardman/kaggen',
    license='MIT',
    packages=find_packages(),
    keywords=['deep learning', 'template', 'pipeline'],
    install_requires=requirements,
)
