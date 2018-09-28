from __future__ import print_function
from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys
from shutil import rmtree

here = os.path.abspath(os.path.dirname(__file__))

VERSION = '0.0.1'
NAME = 'neuralNets'

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')
# long_description = ''


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

# Load the package's __version__.py module as a dictionary.
about = {}
about['__version__'] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    name='neuralNets',
    version=VERSION,
    url='https://github.com/lugq1990/neuralNets',
    license='MIT License',
    author='lugq',
    tests_require=['pytest'],
    install_requires=[
        'matplotlib>=1.4.0',
        'scikit-learn>=0.18',
        'scipy>=0.9',
        'tensorflow>=1.9'
    ],
    # cmdclass={'test': PyTest},
    author_email='gqianglu@outlook.com',
    description='Build deep learning models more efficient based on TensorFlow.',
    long_description=long_description,
    packages=find_packages(exclude=('test',)),
    include_package_data=True,
    platforms='any',
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Deep Learning',
        ],
    extras_require={
        'testing': ['pytest'],
    },
    cmdclass={
        'upload': UploadCommand,
    }
)
