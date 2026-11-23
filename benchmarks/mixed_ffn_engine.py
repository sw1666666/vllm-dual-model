"""Benchmark-only second compiled backbone. Attention, KV and non-MLP modules are shared."""
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as fn
import triton
import triton.language as tl
from torch import nn

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import get_forward_context

CTX=None


@triton.jit
def _scatter(dst,extra,indices,D:tl.constexpr,BLOCK:tl.constexpr):
    j=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    row=tl.load(indices+j)
    at=dst+row*D+col
    tl.store(at,tl.load(at)+tl.load(extra+j*D+col))


def swiglu(x,g,d):
    pair=fn.linear(x,g)
    y=torch.empty((x.shape[0],g.shape[0]//2),device=x.device,dtype=x.dtype)
    torch.ops._C.silu_and_mul(y,pair)
    return fn.linear(y,d)


@torch.library.custom_op("ffn_stage2::split",mutates_args=())
def split(x:torch.Tensor, layer:int)->torch.Tensor:
    s=CTX
    cg,cd,eg,ed,full=s.weights[layer]
    mode,k,diag=s.key
    if mode=="Cref":
        gate,up=fn.linear(x,full).chunk(2,-1)
        return fn.linear(fn.silu(gate[:,:2304])*up[:,:2304],cd)
    c=swiglu(x,cg,cd)
    if mode=="C" or (mode=="M" and not k):
        return c
    if mode=="W":
        return c+swiglu(x,eg,ed)
    if mode=="dense":
        e=swiglu(x,eg,ed)
        return torch.where(s.mask[:x.shape[0],None],c+e,c)
    assert mode=="M" and k>0
    idx=s.indices[:k]
    if diag: ev=s.clock[layer];ev[0].record()
    z=x.index_select(0,idx)
    if diag: ev[1].record()
    e=swiglu(z,eg,ed)
    if diag: ev[2].record()
    _scatter[(k,6)](c,e,idx,1536,256)
    if diag: ev[3].record()
    return c


@split.register_fake
def _(x,layer):
    return torch.empty_like(x)


class Switch(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.i=i

    def forward(self,x):
        return split(x,self.i)


def shallow(m):
    n=copy.copy(m)
    n._modules=m._modules.copy()
    return n


@support_torch_compile(dynamic_arg_dims={"input_ids":0,"positions":-1})
class Backbone(nn.Module):
    def __init__(self,*,base,vllm_config):
        super().__init__()
        self.base=shallow(base)
        layers=[]
        for i,l in enumerate(base.layers):
            v=shallow(l)
            v.mlp=Switch(i)
            assert v.self_attn is l.self_attn
            layers.append(v)
        self.base.layers=nn.ModuleList(layers)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        return self.base.forward(input_ids,positions,intermediate_tensors,inputs_embeds)


class Paths:
    def __init__(self,runner):
        global CTX
        self.runner=runner
        self.weights=[]
        base=runner.get_model().model
        for layer in base.layers:
            mlp=layer.mlp
            gu=mlp.gate_up_proj.weight.detach()
            d=mlp.down_proj.weight.detach()
            g,u=gu.chunk(2,0)
            assert g.shape==u.shape==(8960,1536) and d.shape==(1536,8960)
            self.weights.append((torch.cat((g[:2304],u[:2304])),d[:,:2304].contiguous(),
                                 torch.cat((g[2304:],u[2304:])),d[:,2304:].contiguous(),gu))
        assert len(self.weights)==28 and self.weights[0][2].dtype==torch.float16
        CTX=self
        self.indices=torch.arange(32,device="cuda",dtype=torch.int64)
        self.mask=torch.zeros(32,device="cuda",dtype=torch.bool)
        self.cfg=copy.copy(runner.vllm_config)
        self.cfg.compilation_config=copy.copy(self.cfg.compilation_config)
        import hashlib
        tag=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        self.cfg.compilation_config.cache_dir=str(Path("artifacts/mixed-ffn-v1/cache",tag).resolve())
        self.cfg.compilation_config.traced_files=set()
        with set_current_vllm_config(self.cfg,prefix="split_backbone"):
            self.alt=Backbone(base=base,vllm_config=self.cfg)
        self.wrappers={}
        self.clocks={}

    def prepare(self, keys):
        r=self.runner
        orig=r.model
        self.indices.copy_(torch.arange(32,device="cuda"))
        set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device=r.device),set_current_vllm_config(self.cfg,prefix="split_backbone"):
                for (mode,k,diag),sizes in keys.items():
                    key=(mode,k,diag)
                    self.key=key
                    if key not in self.wrappers:
                        self.wrappers[key]=CUDAGraphWrapper(self.alt,self.cfg,CUDAGraphMode.FULL)
                    wrap=self.wrappers[key]
                    r.model=wrap
                    for p in sorted(sizes,reverse=True):
                        if any(d.num_tokens==p for d in wrap.concrete_cudagraph_entries):continue
                        if diag:
                            self.clock=[[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)] for _ in range(28)]
                            self.clocks[key,p]=self.clock
                        for graph in (False,True):
                            r._dummy_run(p,cudagraph_runtime_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE,
                                uniform_decode=True,force_attention=True,is_graph_capturing=graph)
                        torch.cuda.synchronize()
        finally:
            r.model=orig
            set_cudagraph_capturing_enabled(False)

    def forward(self,key,kwargs):
        ctx=get_forward_context()
        wrap=self.wrappers[key]
        assert ctx.cudagraph_runtime_mode==CUDAGraphMode.FULL and ctx.batch_descriptor in wrap.concrete_cudagraph_entries
        return wrap(**kwargs)
