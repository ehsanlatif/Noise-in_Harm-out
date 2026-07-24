#!/usr/bin/env python3
"""
Prompt-sensitivity ablation (Reviewer T2.9 / R2): does omission dominance reflect
the pipeline or the *terse* SOAP prompt? For a transcript sample we generate SOAP
notes under (A) the paper's TERSE prompt and (B) an INCLUSIVE prompt, then grade
each against the transcript for coverage (omission proxy) and unsupported content
(commissions), and compare A vs B.

SETUP:  pip install anthropic ; export ANTHROPIC_API_KEY=...
RUN:    python experiments/prompt_sensitivity.py --n 40 --model claude-opus-4-8 \
            --transcripts generated_transcripts
Outputs: experiments/out/prompt_sensitivity.csv and _summary.txt
Design note: this uses the TRANSCRIPT as the reference (an inclusive prompt cannot
be penalised for adding transcript-supported facts), isolating the prompt's effect
on omission vs commission counts. For the paper's gold-fact reference, swap the
reference list for the PriMock57 gold facts.
"""
import os, glob, json, argparse, random, re, statistics as st
random.seed(11)

TERSE = """You are a clinical documentation assistant converting a transcript into a TERSE SOAP note.
Be concise: bullet points, fragments, clinical shorthand; a dense summary, not a verbose report.
Omit any section/point not in the transcript. Always include pertinent negatives/positives the
clinician explicitly asked about. Do not add facts beyond the transcript. Output markdown SOAP."""
INCLUSIVE = """You are a clinical documentation assistant converting a transcript into a THOROUGH SOAP note.
Capture ALL clinically relevant information stated in the transcript, including every symptom,
negative, medication, allergy, and plan detail; prefer completeness over brevity while never adding
facts absent from the transcript. Output markdown SOAP."""

def anthropic_text(model, system, user, max_tokens=1500):
    from anthropic import Anthropic
    m=Anthropic().messages.create(model=model,max_tokens=max_tokens,system=system,
        messages=[{"role":"user","content":user}])
    return m.content[0].text

def parse_json(t):
    t=re.sub(r"^```(json)?|```$","",t.strip(),flags=re.M); m=re.search(r"\{.*\}",t,re.S)
    return json.loads(m.group(0)) if m else {}

def grade(model, transcript, note):
    """Return (covered, reference_total, commissions) using the transcript as reference."""
    sys="You compare a SOAP NOTE to the source TRANSCRIPT. Return strict JSON."
    user=(f"TRANSCRIPT:\n{transcript[:6000]}\n\nNOTE:\n{note[:4000]}\n\n"
          "List the clinically relevant facts present in the TRANSCRIPT, then report: "
          '{"reference_total": <#clinically relevant transcript facts>, '
          '"covered_in_note": <# of those present in the NOTE>, '
          '"unsupported_in_note": <# note statements NOT supported by the transcript>}')
    d=parse_json(anthropic_text(model,sys,user,max_tokens=1200))
    return d.get("covered_in_note"),d.get("reference_total"),d.get("unsupported_in_note")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--transcripts",default="generated_transcripts")
    ap.add_argument("--n",type=int,default=40); ap.add_argument("--model",default="claude-opus-4-8")
    a=ap.parse_args(); os.makedirs("experiments/out",exist_ok=True)
    files=[f for f in glob.glob(os.path.join(a.transcripts,"**","*.txt"),recursive=True)]
    random.shuffle(files); files=files[:a.n]
    print(f"prompt-sensitivity on {len(files)} transcripts, model={a.model}")
    import csv
    rows=[]
    for i,f in enumerate(files,1):
        tx=open(f,encoding="utf-8",errors="ignore").read()
        if len(tx)<200: continue
        rec={"file":os.path.relpath(f,a.transcripts)}
        for tag,prompt in [("terse",TERSE),("inclusive",INCLUSIVE)]:
            note=anthropic_text(a.model,prompt,f"TRANSCRIPT:\n{tx[:6000]}")
            cov,ref,uns=grade(a.model,tx,note)
            rec[f"{tag}_cov"]=cov; rec[f"{tag}_ref"]=ref; rec[f"{tag}_uns"]=uns
            if cov is not None and ref: rec[f"{tag}_omit"]=ref-cov
        rows.append(rec)
        if i%5==0: print(f"  {i}/{len(files)}")
    with open("experiments/out/prompt_sensitivity.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    def mean(k):
        v=[r[k] for r in rows if isinstance(r.get(k),(int,float))]; return st.mean(v) if v else float("nan")
    s=(f"n={len(rows)} transcripts\n"
       f"TERSE     omissions/note={mean('terse_omit'):.2f}  commissions/note={mean('terse_uns'):.2f}\n"
       f"INCLUSIVE omissions/note={mean('inclusive_omit'):.2f}  commissions/note={mean('inclusive_uns'):.2f}\n"
       "Interpretation: if INCLUSIVE sharply cuts omissions with few added commissions, omission\n"
       "dominance is partly prompt-induced (report as a caveat); if omissions stay high, it is a\n"
       "pipeline property (strengthens the claim).")
    print("\n"+s); open("experiments/out/prompt_sensitivity_summary.txt","w").write(s)

if __name__=="__main__": main()
