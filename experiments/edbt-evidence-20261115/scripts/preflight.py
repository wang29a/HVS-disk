"""Short compilation checks only; no tuning/search benchmark or graph build."""
import sys,traceback
import common as c
import openimages,compile_native

def main():
    for n in ('logs','commands','guards'):(c.RUN/n).mkdir(parents=True,exist_ok=True)
    # Keep preflight status separate from the live first-priority coordinator.
    c.status=lambda stage,**kw:c.save(c.RUN/'preflight-status.json',dict(stage=stage,**kw))
    import unittest,importlib.util
    spec=importlib.util.spec_from_file_location("edbt_test_protocol",str(c.ROOT/"scripts/test_protocol.py"))
    test_protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(test_protocol)
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_protocol)
    result=unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():raise RuntimeError('protocol unit tests')
    c.run(['clang-format','-style=LLVM','-i',c.ROOT/'scripts/native_search.cpp',c.ROOT/'scripts/native_counts.h'],'format-native-sources',cpus='80,81',timeout=60)
    openimages.compile_executor();compile_native.main()
    c.save(c.RUN/'preflight.json',dict(status='PASS_COMPILED_NOT_RUNTIME_GATED',unit_cases=result.testsRun,native=c.load(c.RUN/'native/validation/build.json'),openimages=c.load(c.RUN/'openimages/validation/build.json')))
if __name__=='__main__':
    try:main()
    except Exception as e:c.save(c.RUN/'preflight.json',dict(status='FAILED',error=str(e)));raise
