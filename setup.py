from distutils.core import setup

setup(name='Karys',
      version='1.0',
      description='Personal implementations of popular ML structures',
      author='Parker Kary',
      author_email='pdkary@gmail.com',
      packages=['data','image_utils','layers','models','plotting','testing','trainers'],
      install_requires=[
        'pandas',
        'tensorflow',
        'tensorflow_addons',
        'tensorflow-gpu==2.9.1',
        'numpy',
        'jupyter',
        'jupyterplot',
        'legitindicators',
        'scikit-learn',
        'dictdiffer',
    ]
)