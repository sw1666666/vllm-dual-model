"""Export independent HF checkpoints; check every layer against masked full SwiGLU."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

OUT = Path("artifacts/static-ffn-v1")
SRC = Path("/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
WIDTHS = (8960, 4480, 2304)


def model(w):
    return SRC if w == 8960 else SRC.parent / f"{SRC.name}-static-ffn-{w}"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csvwrite(path, records):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(records[0]))
        w.writeheader()
        w.writerows(records)


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True, parents=True)
    old = json.loads(Path("artifacts/free-kv-v1/run_config.json").read_text())
    cfg = json.loads((SRC/"config.json").read_text())
    source = SRC/"model.safetensors"
    assert (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"]) == (1536, 28, 8960)
    assert sha(source) == old["large_safetensors_sha256"]
    assert sha(SRC/"tokenizer.json") == old["tokenizer_sha256_both"]
    state = load_file(str(source))
    manifests, checks = {}, []
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(32, 1536, device="cuda", dtype=torch.float16)
    for width in WIDTHS:
        dest = model(width)
        sl = dict(state)
        for i in range(28):
            key = f"model.layers.{i}.mlp."
            for name in ("gate_proj", "up_proj", "down_proj"):
                k = key + name + ".weight"
                sl[k] = (state[k][:, :width] if name=="down_proj" else state[k][:width]).contiguous()
        if width != 8960 and not dest.exists():
            dest.mkdir(exist_ok=False)
            for p in SRC.iterdir():
                if p.is_file() and p.name not in ("config.json","model.safetensors"):
                    shutil.copy2(p, dest/p.name)
            (dest/"config.json").write_text(json.dumps(dict(cfg, intermediate_size=width), indent=2))
            save_file(sl, str(dest/"model.safetensors"), metadata={"format":"pt"})
        disk = load_file(str(dest/"model.safetensors"))
        assert disk.keys()==sl.keys() and all(torch.equal(disk[k],v) for k,v in sl.items())
        assert json.loads((dest/"config.json").read_text()) == dict(cfg, intermediate_size=width)
        assert sha(dest/"tokenizer.json")==old["tokenizer_sha256_both"]
        manifests[width] = dict(path=str(dest),weights=sha(dest/"model.safetensors"),
                                config=sha(dest/"config.json"), tokenizer=sha(dest/"tokenizer.json"))
        for i in range(28):
            key = f"model.layers.{i}.mlp."
            names = [key+s+".weight" for s in ("gate_proj","up_proj","down_proj")]
            g,u,d = [state[n].to("cuda",torch.float16) for n in names]
            a,b,c = [disk[n].to("cuda",torch.float16) for n in names]
            assert a.shape==b.shape==(width,1536) and c.shape==(1536,width)
            mid = F.silu(F.linear(x,g))*F.linear(x,u)
            mid[:,width:]=0
            ref = F.linear(mid,d).float()
            got = F.linear(F.silu(F.linear(x,a))*F.linear(x,b),c).float()
            fp32 = F.linear(F.silu(F.linear(x.float(),a.float()))*F.linear(x.float(),b.float()),c.float())
            err = (ref-got).abs().max().item()
            rel = ((ref-got).norm()/ref.norm()).item()
            cal_abs = (ref-fp32).abs().max().item()
            cal_rel = ((ref-fp32).norm()/fp32.norm()).item()
            cut = F.linear(mid[:,:width].contiguous(),d[:,:width].contiguous()).float()
            cut_err = (cut-got).abs().max().item()
            wide32 = F.silu(F.linear(x.float(),g.float()))*F.linear(x.float(),u.float())
            wide32[:,width:]=0
            wide32 = F.linear(wide32,d.float())
            algebra = ((wide32-fp32).norm()/wide32.norm()).item()
            ulp_bound = torch.finfo(torch.float16).eps*ref.abs().max().item()
            passed = (algebra<1e-5 and err<=2*cal_abs+ulp_bound and rel<=max(2*cal_rel,1e-6)
                      and cut_err<=2*cal_abs+ulp_bound)
            checks.append(dict(width=width,layer=i,gate=list(a.shape),up=list(b.shape),down=list(c.shape),
                               dtype="fp16",max_abs=err,relative_l2=rel,calibration_abs=cal_abs,
                               calibration_relative_l2=cal_rel,same_shape_max_abs=cut_err,
                               fp32_algebra_relative_l2=algebra,ulp_bound=ulp_bound,passed=passed))
            if i in (0,27):print(checks[-1],flush=True)
            assert passed, checks[-1]
        del disk,sl
    csvwrite(OUT/"checks.csv",checks)
    (OUT/"models.json").write_text(json.dumps(manifests,indent=2))


if __name__=="__main__":
    main()
