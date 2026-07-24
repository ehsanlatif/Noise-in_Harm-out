#!/usr/bin/env python3
"""
Diarization Error Rate (Reviewer T2.8 / R1): separate speaker-attribution error
from lexical ASR error, so the "acoustic noise -> ASR -> note" chain is not
confounded by diarization. Computes DER (miss + false-alarm + confusion) per
condition from reference vs. hypothesis RTTMs, with a boundary collar.

Self-contained (numpy only); 2-speaker best-permutation mapping (doctor/patient).

REFERENCE RTTM (build once from PriMock57's separated channels):
  Each consultation ships as separate doctor and patient WAVs. Run VAD on each
  channel (e.g., webrtcvad or silero-vad), label doctor segments "DR" and patient
  segments "PT", and write one reference RTTM per consultation. (These are the
  ground-truth speaker regions; the two-channel setup makes them exact.)
HYPOTHESIS RTTM: your pyannote.audio diarization output per (asr,env,snr) run
  (already produced by the pipeline; see asr/pyannote_textgrid / results RTTMs).

RUN:  python experiments/diarization_der.py --ref REF_RTTM_DIR --hyp HYP_RTTM_DIR --collar 0.25
Outputs: per-file DER and a macro/micro summary; group by SNR/environment via
filename to report DER as a function of noise (the key table for the paper).
"""
import os, glob, argparse, itertools, numpy as np

def read_rttm(path):
    segs=[]
    for ln in open(path):
        p=ln.split()
        if len(p)>=8 and p[0]=="SPEAKER":
            segs.append((float(p[3]), float(p[3])+float(p[4]), p[7]))
    return segs

def label_frames(segs, T, hop=0.01):
    n=int(T/hop)+1; lab=[set() for _ in range(n)]
    for s,e,spk in segs:
        for i in range(max(0,int(s/hop)), min(n,int(e/hop))): lab[i].add(spk)
    return lab

def der_file(ref, hyp, hop=0.01, collar=0.25):
    T=max([e for _,e,_ in ref]+[e for _,e,_ in hyp]+[0.0])
    R=label_frames(ref,T,hop); H=label_frames(hyp,T,hop)
    # collar: drop frames within `collar` of any reference boundary
    keep=np.ones(len(R),bool)
    for s,e,_ in ref:
        for b in (s,e):
            for i in range(max(0,int((b-collar)/hop)),min(len(R),int((b+collar)/hop))): keep[i]=False
    rspk=sorted({s for seg in ref for s in [seg[2]]}); hspk=sorted({s for seg in hyp for s in [seg[2]]})
    best=None
    for perm in itertools.permutations(hspk, len(hspk)):
        mp=dict(zip(hspk, (perm+tuple(rspk))[:len(hspk)]))  # map hyp->ref labels
        miss=fa=conf=ref_speech=0
        for i in range(len(R)):
            if not keep[i]: continue
            r=R[i]; h={mp.get(x,x) for x in H[i]}
            ref_speech+=len(r)
            miss+=len(r-h); fa+=len(h-r)
            conf+=0  # single-label frames -> confusion folded into miss+fa for 2-spk
        tot=ref_speech if ref_speech else 1
        val=(miss+fa+conf)/tot
        if best is None or val<best[0]: best=(val,miss,fa,ref_speech)
    return best  # (DER, miss, fa, ref_speech_frames)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ref",required=True); ap.add_argument("--hyp",required=True)
    ap.add_argument("--collar",type=float,default=0.25)
    a=ap.parse_args()
    refs={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(a.ref,"**","*.rttm"),recursive=True)}
    tot_err=tot_ref=0; per=[]
    for name,hf in {os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(a.hyp,"**","*.rttm"),recursive=True)}.items():
        # match hypothesis to reference by consultation id (prefix before first '_')
        cid=name.split("_")[0]+"_"+name.split("_")[1] if "_" in name else name
        rf=refs.get(name) or next((refs[k] for k in refs if k.startswith(cid)), None)
        if not rf: continue
        der,miss,fa,rs=der_file(read_rttm(rf),read_rttm(hf),collar=a.collar)
        per.append((name,der)); tot_err+=miss+fa; tot_ref+=rs
    if not per: print("No matched ref/hyp RTTM pairs found. Check --ref/--hyp and naming."); return
    macro=float(np.mean([d for _,d in per])); micro=tot_err/max(tot_ref,1)
    print(f"files matched: {len(per)}")
    print(f"macro-avg DER = {macro:.3f}  |  micro-avg DER = {micro:.3f}  (collar={a.collar}s)")
    print("Report DER stratified by SNR/environment (parse from filenames) to separate")
    print("speaker-attribution error from lexical WER, and relabel the vignette accordingly.")

if __name__=="__main__": main()
