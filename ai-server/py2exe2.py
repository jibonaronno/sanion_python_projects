from py2exe import freeze

freeze(console=[{"path":"./consumerClass.py"}],
    windows=[],
    data_files=None,
    zipfile=None,
    options={},
    version_info={})