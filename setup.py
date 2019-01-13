import os.path
import sys
from setuptools import setup, find_packages

sys.path.append(os.path.join(os.path.dirname(__file__), 'receptor'))
from receptor.config import version


install_requires = [
    'numpy',
    'torch',
    'gym>=0.9.1',
    'six',
    'opencv-python',
    'tensorboard',
    'tensorboardX'
]

extras_require = {
    'universe': ['universe>=0.21.3']
}

setup(name='receptor',
      version=version,
      description='Reinforcement Learning framework based on PyTorch and OpenAI Gym',
      url='https://github.com/dbobrenko/receptor',
      author='Dmytro Bobrenko',
      author_email='d.bobrenko@gmail.com',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('receptor')],
      install_requires=install_requires,
      extras_require=extras_require,
      zip_safe=False)
