"""Compile two isolated driver archives using the recorded compiler and source."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from instrument import builder, driver


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8*2**20),b""):
            h.update(data)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main(root, host, old):
    b=root/"build"
    b.mkdir(exist_ok=True)
    proof=read(old/"progress.json")["phase1-executable"]["result"]
    variant=host/"experiments/u-shape-20261113/stage2-v2"
    phase=read(variant/"validation/semantic-proof.json")
    archive=variant/"bin/libpipeann-unified.a"
    source=variant/"variant/unified/hybrid_index.cpp"
    original=Path(proof["binary"])
    src_driver=old/"source/build_merge.cpp"
    for p,h in ((archive,proof["library_sha256"]), (original,proof["sha256"]),
                (src_driver,proof["source_sha256"]), (source,phase["binaries"]["unified"]["source_sha256"])):
        if sha(p)!=h:
            raise RuntimeError("Frozen input changed: "+str(p))
    gen=b/"hybrid_index.cpp"
    gen.write_text(builder(source),encoding="utf-8")
    command=next(c[c.index("g++-11"):] for c in phase["commands"] if "g++-11" in c
                 and "-c" in c and str(source) in c)
    cfg=read(host/"experiments/metric-consistent-20261111/analysis/build.json")
    gen_driver=b/"driver.cpp"
    gen_driver.write_text(driver(src_driver),encoding="utf-8")
    results={}
    include=["-I"+str((root/"source").resolve())]
    with (root/"logs/compile.log").open("w") as log:
        for name in ("off","on"):
            folder=b/name
            folder.mkdir(exist_ok=True)
            obj=folder/"hybrid_index.cpp.o"
            lib=folder/archive.name
            exe=folder/"build-merge"
            flags=["-DHVS_LINK_PROFILE"] if name=="on" else []
            cc=command[:-4]+include+flags+["-c",str(gen),"-o",str(obj)]
            subprocess.run(cc,check=True,stdout=log,stderr=subprocess.STDOUT)
            shutil.copyfile(archive,lib)
            subprocess.run(["ar","r",str(lib),str(obj)],check=True,stdout=log,stderr=subprocess.STDOUT)
            link=["g++-11",*cfg["flags"],"-fno-fast-math","-ffp-contract=off",*flags,str(gen_driver),str(lib),
                  *cfg["link"],"-o",str(exe)]
            subprocess.run(link,check=True,stdout=log,stderr=subprocess.STDOUT)
            results[name]={"binary":str(exe),"sha256":sha(exe),"compile":cc,"link":link}
    (b/"identity.json").write_text(json.dumps(dict(original=proof,binaries=results,
            generated_builder_sha256=sha(gen),generated_driver_sha256=sha(gen_driver)),indent=2)+"\n")
    print(json.dumps({name:v["sha256"] for name,v in results.items()}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","host","old"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    main(a.root,a.host,a.old)
