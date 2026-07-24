#!/usr/bin/env python3
"""Reviewer-requested re-analyses for 'Noise In, Harm Out' (AAAI-27 AISI).
Runs on the study's own per-note result data (results/*.json -> paper_subset.csv).
All analyses are on REAL data; nothing is synthesized.

Addresses: R3 (clustered inference + FDR, rank-stability, weight-sensitivity,
leave-cluster-out), R1 (rank reframing, speech-env exclusion), R4 (robustness).
"""
import itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

PAP = pd.read_csv("experiments/paper_subset.csv")
FULL = pd.read_csv("experiments/full_dataset.csv")
PAP["hi"] = PAP.N1 + PAP.N2            # high-severity (Tier 1+2)
FULL["hi"] = FULL.N1 + FULL.N2
SPEECH_ENVS = {"hospital_reception", "hospital_maternity_ward"}

def per_consult(df, snr, metric):
    """mean of `metric` per consultation at a given snr level (str)."""
    s = df[df.snr.astype(str) == str(snr)].groupby("consult")[metric].mean()
    return s

def paired(df, metric, snr, ref="clean"):
    a = per_consult(df, snr, metric); b = per_consult(df, ref, metric)
    idx = a.index.intersection(b.index)
    return (a[idx] - b[idx]).values                      # paired diffs, one per consultation

def perm_p(d):
    """exact paired sign-flip permutation test (two-sided) on the mean; enumerates 2^n."""
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    if n == 0: return np.nan
    if n <= 20:
        cnt = tot = 0
        for signs in itertools.product((1, -1), repeat=n):
            tot += 1
            if abs((d * np.array(signs)).mean()) >= obs - 1e-12: cnt += 1
        return cnt / tot
    # fallback MC for larger n
    draws = rng.choice((1, -1), size=(20000, n))
    return (np.abs((draws * d).mean(1)) >= obs - 1e-12).mean()

def boot_ci(d, B=10000):
    d = np.asarray(d, float); n = len(d)
    means = d[rng.integers(0, n, size=(B, n))].mean(1)
    return np.percentile(means, [2.5, 97.5])

def bh_fdr(ps):
    ps = np.asarray(ps, float); n = len(ps); order = ps.argsort()
    q = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, ps[i] * n / k); q[i] = prev
    return q

print("="*78); print("E1. CLUSTERED RE-ANALYSIS (cluster-level exact tests) + BH-FDR"); print("="*78)
metrics = [("S","severity-weighted"),("n_om","omissions"),("n_com","commissions"),
           ("tot","total errors"),("hi","high-severity T1+2")]
rows=[]
for snr in ["13","8","-2"]:
    for m,lbl in metrics:
        d = paired(PAP, m, snr); lo,hi = boot_ci(d)
        rows.append(dict(snr=snr, metric=lbl, n=len(d), mean_delta=d.mean(),
                         ci=f"[{lo:.2f}, {hi:.2f}]", p_perm=perm_p(d)))
res = pd.DataFrame(rows)
res["q_FDR"] = bh_fdr(res.p_perm.values)
res["sig_FDR"] = np.where(res.q_FDR < .05, "yes", "NO")
pd.set_option("display.width",130, "display.max_columns",20)
print(res.to_string(index=False, float_format=lambda x:f"{x:.4f}"))
print("\nMILD-NOISE (13 dB) check:",
      res[(res.snr=="13")&(res.metric=="omissions")][["p_perm","q_FDR","sig_FDR"]].to_dict("records"))

print("\n"+"="*78); print("   GEE (Poisson, cluster-robust, groups=consultation) — omissions ~ SNR"); print("="*78)
import statsmodels.api as sm, statsmodels.formula.api as smf
d2 = PAP.copy(); d2["snr"]=pd.Categorical(d2.snr.astype(str), categories=["clean","13","8","-2"])
try:
    g = smf.gee("n_om ~ C(snr)", groups="consult", data=d2,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    print(g.summary().tables[1])
except Exception as e:
    print("GEE failed:", e)

print("\n"+"="*78); print("E2. BOOTSTRAP RANK-STABILITY (cluster bootstrap over consultations)"); print("="*78)
def cluster_boot_ranks(df, groupcol, value="S", B=5000, ascending=True):
    piv = df.pivot_table(index="consult", columns=groupcol, values=value, aggfunc="mean")
    names = list(piv.columns); arr = piv.values; n = arr.shape[0]
    rankcount = {nm: np.zeros(len(names)+1) for nm in names}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = np.nanmean(arr[idx, :], axis=0)
        order = np.argsort(m) if ascending else np.argsort(-m)  # lower burden = rank1
        for r, gi in enumerate(order, 1): rankcount[names[gi]][r] += 1
    return names, rankcount, B
print("\n-- LLM scribes (by mean severity-weighted burden; rank 1 = lowest/best) --")
names,rc,B = cluster_boot_ranks(PAP,"llm",value="S",B=5000)
overall = PAP.groupby("llm").S.mean()
for nm in names:
    p1=rc[nm][1]/B
    print(f"  {nm:35s} meanS={overall[nm]:5.2f}  P(rank1)={p1:.2f}  P(rank2)={rc[nm][2]/B:.2f}  P(rank3)={rc[nm][3]/B:.2f}")
print("\n-- ASR systems by environment: stable best/worst vs volatile mid-field --")
for env in ["outdoor_traffic","hospital_corridor","hospital_reception","hospital_maternity_ward"]:
    sub=PAP[PAP.env==env]
    nm,rcx,Bx=cluster_boot_ranks(sub,"asr",value="S",B=2000)
    mean_=sub.groupby("asr").S.mean().sort_values()
    best=mean_.index[0]; worst=mean_.index[-1]
    pbest=rcx[best][1]/Bx; pworst=rcx[worst][len(nm)]/Bx
    print(f"  {env:24s} best={best:24s} P(best=rank1)={pbest:.2f} | worst={worst:20s} P(worst=last)={pworst:.2f}")

print("\n"+"="*78); print("E3a. WEIGHT-SENSITIVITY of the clean->severe 'doubling'"); print("="*78)
def burden(df, w):  # w = (w1,w2,w3,w4)
    return (w[0]*df.N1+w[1]*df.N2+w[2]*df.N3+w[3]*df.N4)
for name,w in [("linear (4,3,2,1) [paper]",(4,3,2,1)),("convex (8,4,2,1)",(8,4,2,1)),
               ("steep (10,5,2,1)",(10,5,2,1)),("unweighted count (1,1,1,1)",(1,1,1,1)),
               ("high-severity only (1,1,0,0)",(1,1,0,0))]:
    tmp=PAP.copy(); tmp["b"]=burden(tmp,w)
    c=per_consult(tmp,"clean","b").mean(); s=per_consult(tmp,"-2","b").mean()
    print(f"  {name:32s} clean={c:6.2f}  severe={s:6.2f}  ratio={s/c:4.2f}x")

print("\n"+"="*78); print("E3b. SPEECH-BEARING ENVIRONMENT EXCLUSION (omission dominance robustness)"); print("="*78)
for label, envs in [("ALL 4 environments", None),
                     ("EXCL speech-bearing (reception+maternity)", {"outdoor_traffic","hospital_corridor"}),
                     ("ONLY speech-bearing", SPEECH_ENVS)]:
    dd = PAP if envs is None else PAP[(PAP.env.isin(envs)) | (PAP.snr=="clean")]
    om = per_consult(dd,"-2","n_om").mean()-per_consult(dd,"clean","n_om").mean()
    cm = per_consult(dd,"-2","n_com").mean()-per_consult(dd,"clean","n_com").mean()
    print(f"  {label:44s} excess_om={om:.2f} excess_com={cm:.2f} omission_share={100*om/(om+cm):.1f}%")

print("\n"+"="*78); print("E3c. LEAVE-CLUSTER-OUT robustness (clean->severe S doubling & omission share)"); print("="*78)
t1 = set(PAP[PAP.N1>0].consult.unique())
print(f"  consultations contributing any Tier-1 event: {len(t1)} of {PAP.consult.nunique()}")
def headline(df):
    c=per_consult(df,"clean","S").mean(); s=per_consult(df,"-2","S").mean()
    om=per_consult(df,"-2","n_om").mean()-per_consult(df,"clean","n_om").mean()
    cm=per_consult(df,"-2","n_com").mean()-per_consult(df,"clean","n_com").mean()
    return s/c, 100*om/(om+cm)
r_all,sh_all=headline(PAP)
r_noT1,sh_noT1=headline(PAP[~PAP.consult.isin(t1)])
print(f"  ALL 11 consultations:            doubling={r_all:.2f}x  omission_share={sh_all:.1f}%")
print(f"  DROP Tier-1 consultations:       doubling={r_noT1:.2f}x  omission_share={sh_noT1:.1f}%  (n={PAP[~PAP.consult.isin(t1)].consult.nunique()})")
# leave-one-consultation-out range of severe S
loo=[]
for c in PAP.consult.unique():
    loo.append(per_consult(PAP[PAP.consult!=c],"-2","S").mean())
print(f"  leave-one-out severe mean S: range [{min(loo):.2f}, {max(loo):.2f}] (full={per_consult(PAP,'-2','S').mean():.2f})")

print("\n"+"="*78); print("E4. REPLICATION ON FULL 21-CONSULTATION DATA (within-noise dose-response)"); print("="*78)
print(f"  consultations available (noisy): {FULL[FULL.snr!='clean'].consult.nunique()}")
for m,lbl in [("S","severity-weighted"),("n_om","omissions"),("n_com","commissions")]:
    means={snr: FULL[FULL.snr.astype(str)==snr][m].mean() for snr in ["13","8","-2"]}
    print(f"  {lbl:20s} 13dB={means['13']:.2f}  8dB={means['8']:.2f}  -2dB={means['-2']:.2f}")
# clustered test 8 vs -2 within noisy on 21 consultations (paired by consultation)
a=FULL[FULL.snr.astype(str)=='-2'].groupby("consult").n_om.mean()
b=FULL[FULL.snr.astype(str)=='8'].groupby("consult").n_om.mean()
idx=a.index.intersection(b.index); d=(a[idx]-b[idx]).values
print(f"  omissions -2dB vs 8dB across {len(d)} consultations: mean delta={d.mean():.2f}, exact p={perm_p(d):.4f}")
print("\nDONE.")
