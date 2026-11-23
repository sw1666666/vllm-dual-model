"""Paired immediate vs bounded-wait scheduling on the same fixed trace."""
import argparse
import time
from pathlib import Path
from statistics import median

from free_kv_replay import LARGE, Replay, rows, save_rows


class ReadyQueue:
    def __init__(self,target,deadline_ms):
        self.target=target
        self.deadline_ms=deadline_ms

    def reset(self,keys):
        self.next_index=dict.fromkeys(keys,1)
        self.pending={}

    def select(self,data,router):
        now=time.perf_counter()
        active=[]
        pred={}
        for k,i in self.next_index.items():
            if i>=len(data[k]["output_ids"]) or k in self.pending:
                continue
            prediction="L" if router.route(k,i)==LARGE else "S"
            if data[k]["route_for_output"][i]=="L":
                self.pending[k]=(now,prediction)
            else:
                active.append(k)
                pred[k]=prediction
        waits={}
        if self.pending:
            oldest=next(iter(self.pending.values()))[0]
            if len(self.pending)>=self.target or not active or (now-oldest)*1000>=self.deadline_ms:
                for k,(ready,prediction) in self.pending.items():
                    active.append(k)
                    pred[k]=prediction
                    waits[k]=(now-ready)*1000
                self.pending.clear()
        # Preserve cohort slot order in both arms.
        active=[k for k in self.next_index if k in pred]
        when={k:self.next_index[k] for k in active}
        for k in active:
            self.next_index[k]+=1
        return active,pred,when,waits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("artifacts/b32-logits-replay"))
    p.add_argument("--small",default="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--large",default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    args=p.parse_args()
    args.mode="routed_free_kv"
    args.job="queue"
    replay=Replay(args)
    trace=rows(Path("artifacts/free-kv-v1/trace.jsonl"))
    batches=[trace[i:i+32] for i in range(0,len(trace),32)]
    for mode in ("routed_free_kv","routed_exact"):
        replay.mode=mode
        deltas=[]
        for rep in range(-1,5):
            times={}
            for target in ((1,4) if rep%2==0 else (4,1)):
                replay.ready_queue=ReadyQueue(target,12.0)
                results=[]
                for group in batches:
                    record,_=replay.cohort(group)
                    for r in group:
                        seq=[s for s in record["requests"] if s["request_id"]==r["request_id"]]
                        assert [x["output_index"] for x in seq]==list(range(1,len(r["output_ids"])))
                        assert [x["predicted"] for x in seq]==r["route_for_output"][1:]
                    results.append(record)
                assert sum(len(r["requests"]) for r in results)==4091
                assert sum(s["heavy"] for r in results for s in r["steps"])==384
                assert sum(s["catchup"] for r in results for s in r["steps"])==(2705 if mode=="routed_exact" else 0)
                times[target]=sum(r["seconds"] for r in results)
                if rep>=0:
                    save_rows(args.out/f"queue-{mode}-{target}-{rep}.jsonl",results)
            if rep>=0:
                deltas.append(times[1]-times[4])
        if mode=="routed_free_kv" and median(deltas)<=0:
            break


if __name__=="__main__":
    main()
