Import('env', 'tss_sources')
import os

tss_library = env.AddLibrary('tss', tss_sources)

# install headers
def add_headers(PREFIX, env, dirname, fnames):
    for f in fnames:
        if f.endswith((".h",".hxx",".hpp",".cuh")):
            src=os.path.join(dirname,f)
            prefix=os.path.join(".", os.path.relpath(dirname, dir_path))
            #env.AddStagedHeaders([src],prefix=os.path.join(PREFIX,prefix))
            env.AddHeaders([src],prefix=os.path.join(PREFIX,prefix), stage=True)

dir_path = Dir('src').srcnode().abspath
os.walk(dir_path, lambda x,y,z: add_headers("tss",x,y,z), env)

