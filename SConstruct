import sconsutils
env = Environment()
sconsutils.generate(env)

env.Append(CXXFLAGS=['-O2', '-std=c++14', '-Wall', '-Wno-unused-parameter', '-Wno-unused-variable', '-Wno-unused-local-typedefs', '-Wno-unused-function', '-Wno-deprecated-declarations', '-Wno-strict-aliasing', '-Wno-int-in-bool-context', '-Werror=unused-parameter', '-ggdb', '-fPIC', '-rdynamic', '-fno-omit-frame-pointer'])
# uncomment this line and comment the one above for a debug build
#env.Append(CXXFLAGS=['-O0', '-ggdb', '-std=c++14', '-Wall', '-Wno-unused-parameter', '-Wno-unused-variable', '-Wno-unused-local-typedefs', '-Wno-unused-function', '-Wno-deprecated-declarations', '-Wno-strict-aliasing', '-Wno-int-in-bool-context', '-ggdb', '-fPIC', '-rdynamic'])
env.Append(LDMODULEFLAGS=['-rdynamic'])

version = ARGUMENTS.get('VERSION', 0)
Export('version')

env_avx = env.Clone(BUILDDIR='avx')
env_avx.Append(CXXFLAGS=['-mavx'])

tss_sources = '''
src/core/communicator.cxx
src/core/covariance_estimator.cxx
src/core/free_energy_estimator.cxx
src/core/history.cxx
src/core/replica.cxx
src/core/tss.cxx
src/core/visits.cxx
src/core/window.cxx

src/util/distributions.cxx
src/util/eigen.cxx
src/util/randgen.cxx
src/util/util.cxx

src/configuration.cxx
'''.split()

Export('env', 'env_avx', 'tss_sources')


env.SConscript('SConscript', variant_dir=env['BUILDDIR'], duplicate=0)
env_avx.SConscript('SConscript_avx', variant_dir=env['BUILDDIR'], duplicate=0)
env_avx.SConscript('SConscript_python', variant_dir=env['BUILDDIR'], duplicate=0)
