"""Reproduce the numbers in the Comparison section of the Normalization post (d = 64, 2,000 inputs, seed 0)."""
import numpy as np
d, N, eps = 64, 2000, 1e-5
rng = np.random.default_rng(0)
Z = rng.standard_normal((N, d))

def l2(X):  return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
def rms(X): return X / np.sqrt((X**2).mean(1, keepdims=True) + eps)
def ln(X):
    mu = X.mean(1, keepdims=True); var = X.var(1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)

def mean_offdiag_cos(Y):
    U = l2(Y); G = U @ U.T
    return (G.sum() - np.trace(G)) / (N * (N - 1))

LN0 = ln(Z)
print(f"{'c':>4} {'angL2':>7} {'angRMS':>7} {'cosL2':>7} {'pred':>7} {'cosRMS':>7} {'cosLN':>8} {'maxRMS-sqrtdL2':>15} {'angLN':>7} {'arctan':>7} {'max|ang-arctan|':>15} {'arctan(c)':>9} {'maxLN(c)-LN(0)':>14}")
for c in [0, 0.5, 1, 2, 3, 5]:
    X = Z + c
    A, R, L = l2(X), rms(X), ln(X)
    diff = np.abs(R - np.sqrt(d) * A).max()
    cosang = (l2(X) * l2(L)).sum(1).clip(-1, 1)
    angA = np.degrees(np.arccos((l2(X) * l2(A)).sum(1).clip(-1, 1)))
    angR = np.degrees(np.arccos((l2(X) * l2(R)).sum(1).clip(-1, 1)))
    ang = np.degrees(np.arccos(cosang))
    mu, sd = X.mean(1), X.std(1)
    pred = np.degrees(np.arctan(np.abs(mu) / sd))
    print(f"{c:>4} {angA.mean():7.2f} {angR.mean():7.2f} {mean_offdiag_cos(A):7.4f} {c*c/(1+c*c):7.4f} {mean_offdiag_cos(R):7.4f} {mean_offdiag_cos(L):8.4f} {diff:15.2e} {ang.mean():7.2f} {pred.mean():7.2f} {np.abs(ang-pred).max():15.2e} {np.degrees(np.arctan(c)):9.2f} {np.abs(L-LN0).max():14.2e}")
print("LN norm/sqrt(d) range:", (np.linalg.norm(LN0,axis=1)/np.sqrt(d)).min(), (np.linalg.norm(LN0,axis=1)/np.sqrt(d)).max())
print("LN max |sum|:", np.abs(LN0.sum(1)).max())
print("mean |mu|/sigma at c=0:", (np.abs(Z.mean(1))/Z.std(1)).mean(), " 1/sqrt(d)=", 1/np.sqrt(d), " sqrt(2/pi)/sqrt(d)=", np.sqrt(2/np.pi)/np.sqrt(d))
