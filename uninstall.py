import subprocess

# Get the list of installed packages
installed_packages = subprocess.check_output(['pip3', 'list']).decode('utf-8').split('\n')

# Uninstall each package one by one
for package in installed_packages[2:-1]:
    package_name = package.split(' ')[0]
    print(package_name)
    if package_name != 'pip':
        subprocess.call(['pip3', 'uninstall', '-y', package_name])
