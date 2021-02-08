# import os.path
# from os import listdir
# import re
from numpy.distutils.core import setup


# scripts = ['seispy/' + i for i in listdir('seispy/')]

setup(
    name='seispy',
    version='0.3',
    description='Python tools for seismic data analysis',
    author='Xiaotao Yang',
    author_email='stcyang@gmail.com',
    maintainer='Xiaotao Yang',
    maintainer_email='stcyang@gmail.com',
    url='https://github.com/xtyangpsp/SeisPy',
    classifiers=[
        'Development Status :: Beta',
        'License :: MIT License',
        'Programming Language :: Python :: >=3.6'],
    install_requires=['numpy', 'scipy', 'pandas','obspy','pyasdf','numba'],
    python_requires='>=3.6',
    packages=['seispy']) #scripts=scripts
