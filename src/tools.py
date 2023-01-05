import pip


def import_or_install(package_name, package_import_name):
    try:
        __import__(package_import_name)
        print(f'{package_name} already installed')
    except ImportError:
        print(f'Installing {package_name}')
        pip.main(['install', package_name])
