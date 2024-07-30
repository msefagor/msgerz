from setuptools import setup, find_packages

setup(
    name='msgerz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'pandas',
        'openpyxl',
        'vaderSentiment',
        'afinn',
        'googletrans==4.0.0-rc1',
        'textblob',
    ],
    entry_points={
        'console_scripts': [
            'msgerz=msgerz.sentiment_analyzer:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A sentiment analysis library that uses BERT, VADER, Afinn, and TextBlob with keyword-based scoring.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/msgerz',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
