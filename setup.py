from setuptools import setup, find_packages

setup(name='libadver',
      version='0.1',
      description='Package for adversarial attack',
      url='http://github.com/selous/libadver',
      author='Tao Zhang',
      author_email='lrhselous@nuaa.edu.cn',
      license='Anti 996',
      #packages=['vistools'],
      packages = find_packages(),
      include_package_data = True,
      zip_safe=False)
