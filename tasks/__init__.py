from invoke import Collection

from . import code, test


ns = Collection()
ns.add_collection(Collection.from_module(code))
ns.add_collection(Collection.from_module(test))
