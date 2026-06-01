Here is the intuition behind these ideas in resource allocation and networking.

---

# 1. Fairness (general idea)

Fairness means:

> “How should we divide limited resources among multiple users in a way that is considered reasonable?”

Suppose users receive allocations:

[
x = (x_1, x_2, \dots, x_n)
]

where (x_i) is the resource given to user (i).

Different fairness notions answer different questions:

* Should everyone get the same?
* Should efficiency matter more?
* Should weak users be protected?
* Should gains be proportional?

There is no single “correct” fairness definition.

---

# 2. Pareto Efficiency

An allocation is **Pareto efficient** if:

> You cannot improve one user without hurting another.

Formally, allocation (x) is Pareto efficient if there is no other feasible allocation (y) such that:

[
y_i \ge x_i \quad \forall i
]

and

[
y_j > x_j \quad \text{for at least one } j
]

Meaning:

* everyone is at least as well off,
* and someone is strictly better off.

If such a (y) exists, then (x) was wasteful or inefficient.

---

# 3. Pareto Frontier

The **Pareto frontier** (or Pareto boundary) is:

> The set of all Pareto efficient allocations.

Geometrically:

* feasible allocations form a region,
* the “outer edge” of best trade-offs is the Pareto frontier.

x_1+x_2=C

For example, if two users share capacity (C), every efficient allocation lies on:

[
x_1 + x_2 = C
]

Any point inside the region wastes capacity.

Important insight:

* **Pareto efficiency does NOT imply fairness.**

Example:

* ((100,0)) may be Pareto efficient,
* but clearly unfair.

So fairness notions are usually methods for selecting *one* point on the Pareto frontier.

---

# 4. Min-Max Fairness

Min-max fairness prioritizes the *worst-off* users.

Idea:

> Increase the smallest allocation first.
> No user can increase their allocation without decreasing someone with an equal or smaller allocation.

It tries to “equalize suffering.”

Example:

Capacity (=10)

* ((5,5)) is min-max fair.
* ((9,1)) is not.

---

# 5. Progressive Filling / Lexicographic Max-Min Fairness

The standard way to compute min-max fairness is **progressive filling**.

Procedure:

1. Start with everyone at (0).
2. Increase all allocations equally.
3. When a user hits a constraint, freeze them.
4. Continue increasing the remaining users.

This produces the **lexicographically maximal** allocation.

Lexicographic idea:

* maximize the smallest allocation,
* then maximize the second smallest,
* then the third, etc.

So allocations are compared like sorted dictionaries.

Example:

Compare:

[
(2,4,8)
\quad \text{vs} \quad
(3,3,8)
]

Sort them:

[
(2,4,8), \quad (3,3,8)
]

Since (3>2), the second allocation is preferred because its worst user is better off.

---

# 6. Proportional Fairness

Proportional fairness balances:

* efficiency,
* and fairness.

Allocation (x^*) is proportionally fair if for any feasible (y):

[
\sum_i \frac{y_i - x_i^*}{x_i^*} \le 0
]

Interpretation:

> The total proportional gain of users cannot be positive.

So any improvement for some users causes larger proportional losses for others.

Equivalent optimization:
\

This utility strongly rewards helping small users while still valuing efficiency.

Why logs?

* small allocations gain a lot from small increases,
* large allocations gain little.

So proportional fairness sits between:

* pure efficiency,
* and strict equality.

---

# 7. Alpha Fairness

(\alpha)-fairness is a *family* of fairness objectives that unifies many fairness notions.

The utility function is:

U(x)=\sum_i \frac{x_i^{1-\alpha}}{1-\alpha}

(for (\alpha \ne 1))

and for (\alpha=1):

[
U(x)=\sum_i \log(x_i)
][
U(x)=
\begin{cases}
\displaystyle \sum_i \frac{x_i^{1-\alpha}}{1-\alpha}, & \alpha \in \mathbb{R}\setminus{1} [1.2em]
\displaystyle \sum_i \log(x_i), & \alpha = 1
\end{cases}
]


Different (\alpha) values produce different fairness notions.

---

## Important special cases

### (\alpha = 0): Throughput maximization

[
U(x)=\sum_i x_i
]

Only efficiency matters.

Can be very unfair.

---

### (\alpha = 1): Proportional fairness

[
U(x)=\sum_i \log(x_i)
]

Balances fairness and efficiency.

---

### (\alpha = 2): More fairness-oriented

Penalizes inequality more strongly.

---

### (\alpha \to \infty): Min-max fairness

As (\alpha) becomes very large:

> The optimization increasingly prioritizes the smallest allocations.

In the limit:

[
\alpha \to \infty
]

the solution approaches lexicographic max-min fairness.

This is why people say:

> “High-(\alpha) fairness approximates min-max fairness.”

Large (\alpha) heavily punishes tiny allocations, forcing the optimizer to protect the weakest users first.

---

# 8. How everything ties together

These concepts form a hierarchy.

---

## Step 1: Feasibility

First, determine all feasible allocations.

Example:

[
x_1+x_2 \le C
]

---

## Step 2: Pareto efficiency

Remove wasteful points.

Keep only the Pareto frontier.

---

## Step 3: Choose a fairness criterion

Different fairness notions pick different Pareto-efficient points:

| Criterion               | Main Goal                       |
| ----------------------- | ------------------------------- |
| Throughput maximization | Maximize total utility          |
| Proportional fairness   | Balance efficiency and fairness |
| Min-max fairness        | Protect weakest users           |
| Lexicographic max-min   | Strict ordered protection       |
| (\alpha)-fairness       | Smooth continuum between them   |

So:

[
\text{Fairness notion}
\Rightarrow
\text{specific point on Pareto frontier}
]

---

# Intuition summary

* **Pareto efficiency** says: “don’t waste resources.”
* **Pareto frontier** is the set of best feasible trade-offs.
* **Min-max fairness** protects the weakest users.
* **Progressive filling** computes min-max fairness incrementally.
* **Proportional fairness** balances efficiency and equality.
* **(\alpha)-fairness** unifies all these ideas in one parameterized framework.
* Large (\alpha) increasingly behaves like min-max fairness.

The linked notes by Vangelis Paschos develop these ideas from optimization and networking perspectives.

Classical max-flow problems are usually formulated as Linear Programs (LPs), because both the objective and the constraints are linear, e.g.

\max \sum_{(s,v)\in E} x_{sv}

However, proportional fairness introduces logarithmic utility terms such as:

\max \sum_{(u,v)\in E} \log(x_{uv})

The logarithm is nonlinear, so the problem is no longer a linear program. Instead, we move to **convex optimization**, because the log function is concave and can still be optimized efficiently with convex solvers.

This is useful because many fairness objectives can be written as convex optimization problems, allowing us to reuse the same optimization framework while only changing the objective function.

True min-max fairness is not naturally expressed as a smooth convex objective. Instead, we approximate it using the (\alpha)-fair utility family:

\max \sum_i \frac{x_i^{1-\alpha}}{1-\alpha}

As (\alpha) becomes very large, this objective increasingly prioritizes the smallest allocations and converges toward min-max fairness. In practice, this allows us to approximate min-max fairness using the same convex optimizer used for proportional fairness, simply by increasing (\alpha).
