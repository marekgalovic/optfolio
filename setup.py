from setuptools import setup, find_packages

setup(name='optfolio',
      version='0.0.1',
      description='Portfolio Optimization using EA',
      author='Marek Galovic',
      author_email='galovic.galovic@gmail.com',
      license='',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          'numpy',
          'numba',
      ],
)
