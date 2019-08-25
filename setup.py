from setuptools import setup

setup(
    name='gpt-2-tensorflow2.0',
    version='beta',
    packages=['utils', 'layers'],
    url='https://github.com/akanyaani/gpt-2-tensorflow2.0',
    license='MIT',
    author='Abhay Kumar',
    author_email='akanyaani@gmail.com',
    description='', install_requires=['click', 'tqdm', 'tensorflow', 'numpy', 'ftfy', 'sentencepiece']
)
