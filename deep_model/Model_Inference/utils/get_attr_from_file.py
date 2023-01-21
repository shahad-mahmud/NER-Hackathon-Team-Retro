from importlib import import_module
def get_attr_file(name):

    p, m = name.rsplit('.', 1)

    mod = import_module(p)
    met = getattr(mod, m)
    return met