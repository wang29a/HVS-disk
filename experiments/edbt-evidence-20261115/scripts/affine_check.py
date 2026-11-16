"""Additional primitive gate before any density timing; no graph mutation."""
import common as c

def main():
    root=c.RUN/'relaxed';src=c.ROOT/'scripts/test_relaxed_affine.cpp';binary=root/'bin/test-affine'
    c.status=lambda stage,**kw:c.save(root/'validation/affine-status.json',dict(stage=stage,**kw))
    c.run(['clang-format','-style=LLVM','-i',src],'format-affine-check',cpus='80,81',timeout=60)
    c.run(['g++-11','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-UNDEBUG','-I'+str(c.ROOT/'scripts'),'-I'+str(c.U/'scripts'),src,'-o',binary],'compile-affine-check',cpus='80,81',timeout=120)
    logs=[]
    for rho in (1,1.025,1.05):
        log=c.run([binary],'affine-grid-'+str(rho),cpus='80,81',timeout=60,extra=dict(HVS_UNIFIED_RHO=str(rho)));logs.append(log.read_text())
    c.save(root/'validation/affine.json',dict(status='PASS',logs=logs,binary=c.sha(binary),source=c.sha(src),scope='independent affine-vs-direct-fusion grid; ambiguous rounded random boundaries counted separately; exact zero-slope closure checked'))
if __name__=='__main__':main()
