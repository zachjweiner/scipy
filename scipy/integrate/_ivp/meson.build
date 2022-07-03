py3.extension_module('base',
  cython_gen.process('base.pyx'),
  c_args: cython_c_args,
  include_directories: inc_np,
  dependencies: py3_dep,
  install: true,
  subdir: 'scipy/integrate/_ivp'
)

py3.extension_module('rk',
  cython_gen.process('rk.pyx'),
  c_args: cython_c_args,
  include_directories: inc_np,
  dependencies: py3_dep,
  install: true,
  subdir: 'scipy/integrate/_ivp'
)

py3.install_sources([
    '__init__.py',
    'bdf.py',
    'common.py',
    'dop853_coefficients.py',
    'ivp.py',
    'lsoda.py',
    'radau.py',
  ],
  pure: false,
  subdir: 'scipy/integrate/_ivp'
)

subdir('tests')
