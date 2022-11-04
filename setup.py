from setuptools import setup, find_namespace_packages

setup(name='Karys',
      version='1.0',
      description='Personal implementations of popular ML structures',
      author='Parker Kary',
      author_email='pdkary@gmail.com',
      packages=[
          'karys', 'karys.data', 'karys.image_utils', 'karys.layers',
          'karys.models', 'karys.plotting', 'karys.testing', 'karys.trainers'
      ],
      install_requires=[
          'pandas',
          'tensorflow',~
          'tensorflow_addons',
          'tensorflow-gpu==2.9.1',
          'numpy',
          'jupyter',
          'jupyterplot',
          'legitindicators',
          'scikit-learn',
          'dictdiffer',
      ])
