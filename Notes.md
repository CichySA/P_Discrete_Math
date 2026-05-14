Min cut dual interpretation:
f_uv and f_vu - two variables per edge. 0 if there is a *directed* edge
d_uv - one variable per edge. Either the edge is in the cut or isn't.

f_uv <= c_uv - constraint on capacity
in dual, d_uv isn't constrained by c_uv. it's a constant in the objective function min sum c_uv * d_uv, (u, v) in E

new variable - z_v, v in V / {s, t} - variable per non-terminal nodes
if z_v is connected to Source s (in set S) after cut, z_v = 1, 0 if not

constraints:
d_uv - z_u + z_v >= 0
interpretation: if z_u is in S, then either z_v must be in S OR d_uv is a cut edge (counted in min cut)

d_sv + z_v >= 1
d_ut - z_u >= 0
interpretation:
extension of the first constraints: since s is always in S and t in T, this constraint allows edges connected to s or t to be cut and be counted

d_st >= 1
interpretation:
this edge must always be cut. s and t must not be in the same set

interpretation of constraints:
note that they guarantee that if z_v and z_u are in different sets, d_uv *must* be in min cut set
the constraints allow for edges (u, v) not in the cut to be counted, but since this is a minimization problem and d_uv is >=0, it doesn't matter

not that d >= 0 and z is Real. This slackening of variables allows the program to be Linear, not an Integer or non-linear

What forces the optimal solution to take values 0 or 1 for d?

KKT conditions:
Stationarity:
Lagrangian defines stationarity of primal and dual:
D f(x) + l * D h(x) + k * D g(x) = D L in 0
L = 0 if x*
in min cut:
f(x) = sum c * d
g(x) = sum(d_uv - z_u + z_v)
multiplier: f
L = sum(c * d) +  sum f_uv * (z_v - z_u - d_uv) = sum d (c - f) + sum f ( z_v - z_ u)
L / d_uv = sum c - sum f = sum (c - f) = 0
L / f = z_v - z_u - d_uv = 0 => d_uv = z_v - z_u

L / z_v = D sum f_uv * z_v + sum f_vw * z_v / D z_v
= sum f_uv - sum f_vw = 0

