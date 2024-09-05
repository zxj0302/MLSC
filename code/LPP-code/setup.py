from setuptools import setup

setup(name='Gmatch',
      version='0.1',
      description='Code for the subgraph mathcing/containment decision, using NN approachs',
      license='MIT',
      package_dir={'': 'src/'},
      packages=['gmatch'],
      zip_safe=False)