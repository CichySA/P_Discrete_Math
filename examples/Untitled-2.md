Then include the edges whose **edge-flow values** you want to treat fairly.

For a directed graph (G=(V,E)), with edge flow (x_{uv}) on edge ((u,v)\in E):

[
\boxed{
\max \sum_{(u,v)\in E} \log(x_{uv})
}
]

Usually add positivity:

[
x_{uv} > 0 \qquad \forall (u,v)\in E
]

because (\log(0)) is undefined.

If you only want fairness over a subset of edges, say (E_f\subseteq E):

[
\boxed{
\max \sum_{(u,v)\in E_f} \log(x_{uv})
}
]

where, for example,

[
E_f={(u,v)\in E: u\ne s,\ v\ne t}
]

would exclude edges directly leaving (s) or entering (t).

Full single-commodity version:

[
\max \sum_{(u,v)\in E_f}\log(x_{uv})
]

subject to

[
0 < x_{uv}\le c_{uv}\qquad \forall (u,v)\in E
]

[
\sum_{(u,v)\in E}x_{uv}
=======================

\sum_{(v,w)\in E}x_{vw}
\qquad
\forall v\in V\setminus{s,t}
]

[
F=\sum_{(s,v)\in E}x_{sv}
=========================

\sum_{(u,t)\in E}x_{ut}
]

So the answer is: use (\sum_{(u,v)\in E_f}\log(x_{uv})), where (E_f) is the set of edges you want fair edge usage across.

Because the logarithm naturally values **relative (percentage) improvements** rather than absolute improvements.

That is exactly what proportional fairness is about.

---

# The optimization

Proportional fairness comes from solving:

\max \sum_i \log(x_i)

where (x_i) is the allocation to user (i).

---

# Key property of the logarithm

The derivative is:

[
\frac{d}{dx}\log(x)=\frac{1}{x}
]

This means:

* small (x_i) → very large marginal utility,
* large (x_i) → small marginal utility.

So the optimizer strongly prefers helping poorly served users.

But the deeper reason is:

> log converts multiplicative/proportional changes into additive ones.

---

# Small-change intuition

Suppose a user changes from:

* (x_i)
* to (x_i+\Delta x_i)

Then:

[
\log(x_i+\Delta x_i)-\log(x_i)
]

For small changes, using Taylor expansion:

[
\log(x_i+\Delta x_i)-\log(x_i)
\approx
\frac{\Delta x_i}{x_i}
]

This is the crucial step.

Notice what appears:

[
\frac{\Delta x_i}{x_i}
]

which is the **relative/proportional change**.

Examples:

* from (1\to2): (+100%)
* from (100\to101): (+1%)

The log utility considers the first much more important.

---

# Why this creates proportional fairness

Suppose we are at an optimal allocation (x^*).

Any feasible perturbation (\Delta x_i) must satisfy:

[
\sum_i
\left(
\log(x_i^*+\Delta x_i)-\log(x_i^*)
\right)
\le 0
]

Using the approximation above:

[
\sum_i \frac{\Delta x_i}{x_i^*}\le 0
]

which is exactly the proportional fairness condition.

Interpretation:

> the total proportional gains cannot exceed the total proportional losses.

So:

* a user gaining (10%)
  must cause others to lose at least (10%) collectively.

---

# Intuitive example

Suppose two users have allocations:

[
(1,100)
]

Adding 1 unit to the poor user:

[
1 \to 2
]

doubles their allocation:

* (+100%)

Adding 1 unit to the rich user:

[
100 \to 101
]

is only:

* (+1%)

The log objective strongly prefers helping the first user.

So proportional fairness naturally balances:

* efficiency,
* and relative equity.

---

# Another geometric interpretation

The log function is:

* increasing,
* but concave.

Concavity means:

> diminishing returns.

Each extra resource unit gives:

* large benefit to small allocations,
* small benefit to already-large allocations.

This prevents the optimizer from concentrating resources too aggressively.

---

# Why not linear utility?

Compare:

### Linear objective

[
\max \sum_i x_i
]

Marginal utility:

[
\frac{d}{dx}x = 1
]

Every extra unit is valued equally regardless of who gets it.

So the optimizer only chases efficiency.

---

### Log objective

[
\frac{d}{dx}\log(x)=\frac{1}{x}
]

Now marginal value depends inversely on current allocation.

That is what embeds fairness.

---

# Deep connection

Logarithms are special because they transform:

[
\log(ab)=\log a+\log b
]

So maximizing sum of logs is equivalent to maximizing a product:

[
\max \sum_i \log(x_i)
\quad\Longleftrightarrow\quad
\max \prod_i x_i
]

This means the optimizer tries to maximize the geometric mean of allocations.

The geometric mean naturally penalizes imbalance much more than the arithmetic mean.

That is another reason proportional fairness emerges.
