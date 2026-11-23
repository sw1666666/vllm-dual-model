"""Independent, untimed profiler replay of the actual split backbone at each k."""
from types import SimpleNamespace

from free_kv_replay import Replay,torch
from mixed_ffn_engine import CUDAGraphMode
from mixed_ffn_replay import Driver,TRACE,rows,sizes,save


@torch.inference_mode()
def main():
    tr=rows(TRACE);groups=[tr[i:i+32] for i in range(0,len(tr),32)]
    p=Driver(Replay(SimpleNamespace(mode="large",large="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",job="ops"))).paths
    keys=sizes(groups);r=p.runner;orig=r.model
    p.prepare({("M",0,False):{32}})
    evidence=[]
    try:
        r.model=p.wrappers["M",0,False]
        for k in sorted(keys):
            n=max(keys[k]);p.key=("M",k,False)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
                r._dummy_run(n,cudagraph_runtime_mode=CUDAGraphMode.NONE,uniform_decode=True,force_attention=True)
            gemms=[e.input_shapes[:2] for e in prof.events() if e.name=="aten::mm"]
            for a,b,expected in (([n,1536],[1536,4608],28),([n,2304],[2304,1536],28),
                                 ([k,1536],[1536,13312],28 if k else 0),([k,6656],[6656,1536],28 if k else 0)):
                count=gemms.count([a,b])
                assert count==expected,(k,a,b,count)
                evidence.append(dict(k=k,base_padded=n,A=a,B=b,calls=count))
            if not k:assert not any(13312 in b or 6656 in b for a,b in gemms)
    finally:r.model=orig
    save("gemm_operands.json",evidence)


if __name__=="__main__":main()
