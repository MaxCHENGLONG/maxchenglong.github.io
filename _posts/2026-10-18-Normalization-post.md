---
published: True
title: "Normalization in Transformers"
description: "L2 normalization and RMSNorm preserve the direction of a hidden vector; LayerNorm rotates it."
---

## Introduction

Every Transformer layer normalizes its hidden vectors, and in practice this is done by one of two operations: LayerNorm[^ln], used in the original Transformer[^transformer], GPT-2[^gpt2] and BERT[^bert], or RMSNorm[^rmsnorm], used in LLaMA[^llama] and most recent open LLMs. This post compares them with a third, simpler operation, L2 normalization. L2 normalization is not a normalization layer inside Transformers; it appears elsewhere, for example in embedding retrieval and contrastive learning, where only the direction of a vector matters. We include it as the reference point because it does the least possible, dividing a vector by its Euclidean norm and nothing else, and because RMSNorm will turn out to be exactly L2 normalization followed by a constant scale.

All three divide a vector by some measure of its size, so they are easy to treat as interchangeable. They are not, and the difference comes down to one question: **does the normalization preserve the direction of the input?**

L2 normalization and RMSNorm do: they only change the length of a vector. LayerNorm does not: it subtracts the mean before rescaling, and this rotates the vector by an angle that depends on the ratio of its mean to its standard deviation. This post makes that difference precise. We define the three operations, derive the relations between them and the angle by which LayerNorm rotates its input, and then verify the main claims numerically, including what the rotation buys LayerNorm (invariance to a shared shift) and what it costs (a distorted input).

## Definitions
Throughout this post, let $\bm{x} = (x_1, \dots, x_d)^\top \in \mathbb{R}^{d}$ be the hidden vector of a single token, where $d$ is the hidden dimension. Let $\bm{1} \in \mathbb{R}^{d}$ denote the all-ones vector, $\odot$ element-wise multiplication, and $\epsilon > 0$ a small constant added for numerical stability. All three operations act on each token independently along the feature dimension; none of them uses statistics computed across the batch or the sequence. Concretely, for a hidden state of shape $(B, T, d)$, the statistics are computed over the last dimension $d$ only, separately for each of the $B \times T$ token vectors, so the output for one token depends only on that token's own $d$ entries.

### L2 normalization
L2 normalization rescales $\bm{x}$ so that its Euclidean norm equals one:

$$
\mathrm{L2}(\bm{x}) = \frac{\bm{x}}{\lVert \bm{x} \rVert_2},
\qquad
\lVert \bm{x} \rVert_2 = \sqrt{\sum_{i=1}^{d} x_i^2}.
$$

Element-wise, the $i$-th output is $x_i / \lVert \bm{x} \rVert_2$. In practice the denominator is clamped to avoid division by zero, e.g. $\bm{x} / \max(\lVert \bm{x} \rVert_2, \epsilon_{\mathrm{L2}})$, where $\epsilon_{\mathrm{L2}}$ is a much smaller constant than the $\epsilon$ used by LayerNorm and RMSNorm below (around $10^{-12}$ in PyTorch).

L2 normalization has no learnable parameters. It keeps the direction of $\bm{x}$ and discards only its length, so the output always lies on the unit sphere.

### Layer normalization (LayerNorm)

LayerNorm[^ln] standardizes the entries of $\bm{x}$ to zero mean and unit variance, then applies a learnable affine transformation. First compute the mean and variance over the $d$ features:

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i,
\qquad
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 .
$$

Then

$$
\mathrm{LayerNorm}(\bm{x}) = \frac{\bm{x} - \mu \bm{1}}{\sqrt{\sigma^2 + \epsilon}} \odot \bm{\gamma} + \bm{\beta},
$$

or element-wise,

$$
\mathrm{LayerNorm}(\bm{x})_i = \gamma_i \, \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i .
$$

Here $\bm{\gamma} \in \mathbb{R}^{d}$ (gain, initialized to $\bm{1}$) and $\bm{\beta} \in \mathbb{R}^{d}$ (bias, initialized to $\bm{0}$) are learnable, giving $2d$ parameters per layer. The variance uses the biased estimator (division by $d$, not $d-1$), and $\epsilon$ is typically around $10^{-5}$.

LayerNorm involves two steps: **re-centering** (subtracting $\mu$) and **re-scaling** (dividing by $\sigma$). Re-scaling only changes the length of $\bm{x}$, while re-centering also changes its direction. LayerNorm was the standard choice in the original Transformer[^transformer] and in the GPT-2[^gpt2] and BERT[^bert] families.


### RMS normalization (RMSNorm)

RMSNorm[^rmsnorm] simplifies LayerNorm by removing the re-centering step. It divides $\bm{x}$ by its root mean square and applies a learnable gain:

$$
\mathrm{RMS}(\bm{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2},
\qquad
\mathrm{RMSNorm}(\bm{x}) = \frac{\bm{x}}{\sqrt{\mathrm{RMS}(\bm{x})^2 + \epsilon}} \odot \bm{g},
$$

or element-wise,

$$
\mathrm{RMSNorm}(\bm{x})_i = g_i \, \frac{x_i}{\sqrt{\mathrm{RMS}(\bm{x})^2 + \epsilon}} .
$$

The gain $\bm{g} \in \mathbb{R}^{d}$ is learnable and initialized to $\bm{1}$; there is no bias term, so RMSNorm has $d$ parameters per layer.

Without the mean subtraction, RMSNorm is cheaper to compute and keeps the direction of $\bm{x}$ unchanged before the gain is applied. It has become the default normalization in many recent open LLMs, including the LLaMA family[^llama].


| | L2 normalization | RMSNorm | LayerNorm |
|---|---|---|---|
| Formula | $\dfrac{\bm{x}}{\lVert \bm{x} \rVert_2}$ | $\dfrac{\bm{x}}{\sqrt{\mathrm{RMS}(\bm{x})^2 + \epsilon}} \odot \bm{g}$ | $\dfrac{\bm{x} - \mu \bm{1}}{\sqrt{\sigma^2 + \epsilon}} \odot \bm{\gamma} + \bm{\beta}$ |
| Re-centering | No | No | Yes (subtract $\mu$) |
| Re-scaling by | $\lVert \bm{x} \rVert_2$ | $\sqrt{\mathrm{RMS}(\bm{x})^2 + \epsilon}$ | $\sqrt{\sigma^2 + \epsilon}$ |
| Learnable parameters | None | $\bm{g}$ | $\bm{\gamma}, \bm{\beta}$ |
| Parameters per layer | $0$ | $d$ | $2d$ |
| Direction of $\bm{x}$ | Preserved | Preserved (before gain) | Changed by re-centering |
| Output norm (before affine) | $1$ | $\approx \sqrt{d}$ | $\approx \sqrt{d}$ |
{: .bordered}

<style>
table.bordered {
  border-collapse: collapse;
  margin: 1em auto;
}
table.bordered th,
table.bordered td {
  border: 1px solid #999;
  padding: 6px 12px;
  text-align: center;
}
table.bordered td:first-child {
  text-align: left;
}
</style>

## Mathematical analysis

Throughout this section we set $\epsilon = 0$ and drop the affine parameters ($\bm{\gamma} = \bm{g} = \bm{1}$, $\bm{\beta} = \bm{0}$) unless stated otherwise, and we assume $\bm{x}$ is neither zero nor a constant vector, so that $\lVert \bm{x} \rVert_2 > 0$ and $\sigma > 0$. The affine parameters act element-wise after the normalization and do not change any of the arguments below; the role of $\epsilon$ is discussed at the end of the section.

### RMSNorm is a rescaled L2 normalization

The root mean square and the Euclidean norm differ only by a constant factor:

$$
\mathrm{RMS}(\bm{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} = \frac{\lVert \bm{x} \rVert_2}{\sqrt{d}} .
$$

Therefore

$$
\mathrm{RMSNorm}(\bm{x}) = \frac{\bm{x}}{\mathrm{RMS}(\bm{x})} = \sqrt{d} \, \frac{\bm{x}}{\lVert \bm{x} \rVert_2} = \sqrt{d} \; \mathrm{L2}(\bm{x}) .
$$

RMSNorm and L2 normalization produce exactly the same direction and differ only in the length of the output: $\sqrt{d}$ instead of $1$. The factor $\sqrt{d}$ is not cosmetic. It gives the output unit mean square, $\mathrm{RMS}(\mathrm{RMSNorm}(\bm{x})) = 1$, so a typical entry has magnitude around $1$ regardless of the hidden size. After L2 normalization a typical entry has magnitude around $1/\sqrt{d}$, which is about $0.016$ for $d = 4096$. This is why L2 normalization is usually paired with an explicit scale, such as a temperature or a learnable scalar, whenever its output feeds into further computation.

### LayerNorm is RMSNorm applied to the centered vector

Re-centering is a linear map. Define the centering matrix

$$
\bm{P} = \bm{I} - \frac{1}{d} \bm{1} \bm{1}^\top,
\qquad
\bm{P} \bm{x} = \bm{x} - \mu \bm{1} .
$$

$\bm{P}$ is symmetric and idempotent ($\bm{P}^\top = \bm{P}$, $\bm{P}^2 = \bm{P}$) and satisfies $\bm{P} \bm{1} = \bm{0}$. It is the orthogonal projection onto the hyperplane of zero-mean vectors,

$$
\mathcal{H} = \left\lbrace \bm{y} \in \mathbb{R}^{d} : \bm{1}^\top \bm{y} = 0 \right\rbrace .
$$

The variance of $\bm{x}$ is exactly the mean square of the centered vector:

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 = \mathrm{RMS}(\bm{P} \bm{x})^2 .
$$

Substituting this into the definition of LayerNorm gives

$$
\mathrm{LayerNorm}(\bm{x}) = \frac{\bm{P} \bm{x}}{\mathrm{RMS}(\bm{P} \bm{x})} = \mathrm{RMSNorm}(\bm{P} \bm{x}) = \sqrt{d} \; \mathrm{L2}(\bm{P} \bm{x}) .
$$

LayerNorm is therefore a two-stage operation: project $\bm{x}$ onto the zero-mean hyperplane $\mathcal{H}$, then apply RMSNorm. The identity is exact even with $\epsilon$ and the affine parameters, because $\epsilon$ enters both denominators in the same way:

$$
\mathrm{LayerNorm}_{\bm{\gamma}, \bm{\beta}}(\bm{x}) = \mathrm{RMSNorm}_{\bm{\gamma}}(\bm{P} \bm{x}) + \bm{\beta} .
$$

In particular, when $\bm{x}$ already has zero mean, LayerNorm and RMSNorm coincide up to the bias $\bm{\beta}$.

How far apart the two are depends on the mean. Splitting $\bm{x}$ into its centered part and its mean part, $\bm{x} = \bm{P} \bm{x} + \mu \bm{1}$, is an orthogonal decomposition, so

$$
\mathrm{RMS}(\bm{x})^2 = \sigma^2 + \mu^2 .
$$

Hence $\mathrm{RMS}(\bm{x}) \ge \sigma$, with equality if and only if $\mu = 0$. RMSNorm divides by $\sqrt{\sigma^2 + \mu^2}$ and keeps the mean component; LayerNorm divides by $\sigma$ and removes it. Let $\theta \in [0, \pi]$ be the angle between $\bm{x}$ and $\bm{1}$, so that $\cos\theta = \mu / \mathrm{RMS}(\bm{x})$ and $\sin\theta = \sigma / \mathrm{RMS}(\bm{x})$. Then

$$
\mathrm{RMSNorm}(\bm{x}) = \frac{\bm{P} \bm{x}}{\mathrm{RMS}(\bm{x})} + \frac{\mu \bm{1}}{\mathrm{RMS}(\bm{x})} = \sin\theta \; \mathrm{LayerNorm}(\bm{x}) + \cos\theta \; \bm{1} .
$$

Since $\mathrm{LayerNorm}(\bm{x})$ and $\bm{1}$ are orthogonal and both have length $\sqrt{d}$, the RMSNorm output lies on the great circle connecting them, at angle $\theta$ from $\bm{1}$. When $\bm{x}$ is orthogonal to $\bm{1}$ ($\theta = \pi/2$), the two outputs agree. When the mean dominates ($\theta \to 0$ or $\pi$), the RMSNorm output collapses towards $\pm \bm{1}$, while the LayerNorm output is unaffected.

### Direction: preserved by L2 normalization, rotated by LayerNorm

The identities above settle the central question of this post. L2 normalization and RMSNorm divide $\bm{x}$ by a positive scalar, so their outputs are parallel to the input:

$$
\cos\angle\big(\bm{x}, \mathrm{L2}(\bm{x})\big) = \cos\angle\big(\bm{x}, \mathrm{RMSNorm}(\bm{x})\big) = 1 .
$$

LayerNorm divides $\bm{P} \bm{x}$ instead of $\bm{x}$, so its output points along the centered vector. Let $\phi$ be the angle between $\bm{x}$ and $\mathrm{LayerNorm}(\bm{x})$. Using $\bm{x}^\top \bm{P} \bm{x} = \lVert \bm{P} \bm{x} \rVert_2^2$,

$$
\cos\phi = \frac{\bm{x}^\top \bm{P} \bm{x}}{\lVert \bm{x} \rVert_2 \, \lVert \bm{P} \bm{x} \rVert_2} = \frac{\lVert \bm{P} \bm{x} \rVert_2}{\lVert \bm{x} \rVert_2} = \frac{\sigma}{\mathrm{RMS}(\bm{x})},
\qquad
\tan\phi = \frac{\lvert \mu \rvert}{\sigma} .
$$

The rotation is set entirely by the ratio of the mean to the standard deviation. It vanishes only when $\mu = 0$ and approaches $90^\circ$ when the mean dominates. In terms of the angle $\theta$ between $\bm{x}$ and $\bm{1}$ defined above, $\phi = \lvert \pi/2 - \theta \rvert$: LayerNorm rotates $\bm{x}$ within the plane spanned by $\bm{x}$ and $\bm{1}$ until it is orthogonal to $\bm{1}$. The rest of this section shows that the other differences between the three operations follow from this one.

### Invariance properties

**Re-scaling.** For any $\alpha \neq 0$, the three denominators scale by $\lvert \alpha \rvert$: $\lVert \alpha \bm{x} \rVert_2 = \lvert \alpha \rvert \, \lVert \bm{x} \rVert_2$, $\mathrm{RMS}(\alpha \bm{x}) = \lvert \alpha \rvert \, \mathrm{RMS}(\bm{x})$, and the standard deviation of $\alpha \bm{x}$ is $\lvert \alpha \rvert \, \sigma$. Hence, for $f \in \lbrace \mathrm{L2}, \mathrm{LayerNorm}, \mathrm{RMSNorm} \rbrace$,

$$
f(\alpha \bm{x}) = \mathrm{sign}(\alpha) \, f(\bm{x}) .
$$

All three are invariant to positive re-scaling of their input.

**Re-centering.** Since $\bm{P} \bm{1} = \bm{0}$, we have $\bm{P}(\bm{x} + c \bm{1}) = \bm{P} \bm{x}$ for every $c \in \mathbb{R}$, and therefore

$$
\mathrm{LayerNorm}(\bm{x} + c \bm{1}) = \mathrm{LayerNorm}(\bm{x}) .
$$

L2 normalization and RMSNorm do not have this property. Adding $c \bm{1}$ changes $\mu$ and hence the angle $\theta$. As $c \to \pm\infty$, $\cos\theta \to \pm 1$, so $\mathrm{RMSNorm}(\bm{x} + c \bm{1}) \to \pm \bm{1}$ and $\mathrm{L2}(\bm{x} + c \bm{1}) \to \pm \bm{1} / \sqrt{d}$: the output forgets $\bm{x}$ entirely. Shift invariance is the other side of the rotation: LayerNorm can ignore $c \bm{1}$ only because it discards the component of $\bm{x}$ along $\bm{1}$, and discarding that component is exactly what changes the direction.

**Weight-level invariance.** Inside a network the input to a normalization layer is often the output of a linear layer, $\bm{x} = \bm{W} \bm{a}$. Scaling the weights, $\bm{W} \mapsto \alpha \bm{W}$ with $\alpha > 0$, scales $\bm{x}$ and leaves all three outputs unchanged. Adding the same vector $\bm{v}$ to every row of the weight matrix, $\bm{W} \mapsto \bm{W} + \bm{1} \bm{v}^\top$, shifts $\bm{x}$ by $(\bm{v}^\top \bm{a}) \bm{1}$, which only LayerNorm ignores.

| Transformation | L2 normalization | RMSNorm | LayerNorm |
|---|---|---|---|
| Input re-scaling $\bm{x} \mapsto \alpha \bm{x}$, $\alpha > 0$ | ✓ | ✓ | ✓ |
| Input re-centering $\bm{x} \mapsto \bm{x} + c \bm{1}$ | ✗ | ✗ | ✓ |
| Weight re-scaling $\bm{W} \mapsto \alpha \bm{W}$, $\alpha > 0$ | ✓ | ✓ | ✓ |
| Weight re-centering $\bm{W} \mapsto \bm{W} + \bm{1} \bm{v}^\top$ | ✗ | ✗ | ✓ |
{: .bordered}

Zhang and Sennrich[^rmsnorm] hypothesize that re-scaling invariance is what makes LayerNorm effective and that re-centering invariance is dispensable. RMSNorm keeps the former and drops the latter.

### Geometry of the outputs

Each operation maps $\mathbb{R}^{d}$ onto a sphere. Before the affine parameters,

$$
\mathrm{L2}(\bm{x}) \in \mathbb{S}^{d-1},
\qquad
\mathrm{RMSNorm}(\bm{x}) \in \sqrt{d} \, \mathbb{S}^{d-1},
\qquad
\mathrm{LayerNorm}(\bm{x}) \in \sqrt{d} \, \mathbb{S}^{d-1} \cap \mathcal{H},
$$

where $\mathbb{S}^{d-1} = \left\lbrace \bm{y} \in \mathbb{R}^{d} : \lVert \bm{y} \rVert_2 = 1 \right\rbrace$ is the unit sphere. The LayerNorm case follows from

$$
\bm{1}^\top \mathrm{LayerNorm}(\bm{x}) = \frac{\bm{1}^\top \bm{P} \bm{x}}{\sigma} = 0,
\qquad
\lVert \mathrm{LayerNorm}(\bm{x}) \rVert_2^2 = \frac{\lVert \bm{P} \bm{x} \rVert_2^2}{\sigma^2} = \frac{d \, \sigma^2}{\sigma^2} = d .
$$

Each map is onto its set, since every point of the set is mapped to itself. The sets differ in dimension:

- L2 normalization and RMSNorm map onto a $(d-1)$-dimensional sphere. They remove one degree of freedom, the length of $\bm{x}$.
- LayerNorm maps onto the intersection of a sphere of radius $\sqrt{d}$ with a hyperplane through its center, which is a $(d-2)$-dimensional sphere of radius $\sqrt{d}$. It removes two degrees of freedom: the length and the mean.

Low-dimensional cases make the difference concrete. For $d = 2$, $\mathcal{H}$ is the line spanned by $(1, -1)$, and the sphere of radius $\sqrt{2}$ meets it in only two points, so

$$
\mathrm{LayerNorm}(x_1, x_2) = \mathrm{sign}(x_1 - x_2) \, (1, -1) .
$$

The entire two-dimensional input is reduced to a single bit, while L2 normalization and RMSNorm still retain a full circle of directions. For $d = 3$, the LayerNorm output lies on a circle of radius $\sqrt{3}$ in the plane $x_1 + x_2 + x_3 = 0$, whereas RMSNorm covers the whole sphere of radius $\sqrt{3}$. For realistic hidden sizes ($d$ in the thousands), losing one extra dimension is negligible for capacity, so the practical difference between LayerNorm and RMSNorm lies in whether the direction of $\bm{x}$ is preserved, together with the shift invariance that comes from changing it, rather than in expressiveness.

The affine parameters then deform these sets. The gain stretches the sphere into an axis-aligned ellipsoid with semi-axes $\sqrt{d} \, \lvert g_i \rvert$, which for RMSNorm is always centered at the origin. For LayerNorm, the gain maps the $(d-2)$-sphere to an ellipsoid inside the hyperplane $\mathrm{diag}(\bm{\gamma}) \, \mathcal{H}$, and the bias $\bm{\beta}$ translates it away from the origin.

### The role of $\epsilon$

With $\epsilon > 0$, the output norms are slightly smaller than in the analysis above. For RMSNorm,

$$
\lVert \mathrm{RMSNorm}(\bm{x}) \rVert_2 = \sqrt{d} \, \frac{\mathrm{RMS}(\bm{x})}{\sqrt{\mathrm{RMS}(\bm{x})^2 + \epsilon}} < \sqrt{d},
$$

which is essentially $\sqrt{d}$ as long as $\mathrm{RMS}(\bm{x})^2 \gg \epsilon$. Scale invariance therefore holds only approximately. For very small inputs ($\mathrm{RMS}(\bm{x})^2 \lesssim \epsilon$), the operation behaves like the linear map $\bm{x} \mapsto \bm{x} / \sqrt{\epsilon}$ rather than a normalization. This is precisely what keeps the operation numerically stable: without $\epsilon$, the factor $1 / \mathrm{RMS}(\bm{x})$, and with it the gradient, diverges as $\bm{x} \to \bm{0}$, while with $\epsilon$ the factor never exceeds $1 / \sqrt{\epsilon}$. The same applies to LayerNorm with $\sigma^2$ in place of $\mathrm{RMS}(\bm{x})^2$; for a constant input $\bm{x} = c \bm{1}$ the LayerNorm output is simply $\bm{\beta}$. L2 normalization usually clamps the denominator instead, $\bm{x} / \max(\lVert \bm{x} \rVert_2, \epsilon_{\mathrm{L2}})$, with a much smaller constant (around $10^{-12}$ in PyTorch), so it is exactly scale-invariant except in a tiny neighborhood of the origin.

### Summary

The three operations are the same geometric primitive, projection onto a sphere, and the relations between them fit in one line:

$$
\mathrm{RMSNorm}(\bm{x}) = \sqrt{d} \; \mathrm{L2}(\bm{x}),
\qquad
\mathrm{LayerNorm}(\bm{x}) = \mathrm{RMSNorm}(\bm{P} \bm{x}) = \sqrt{d} \; \mathrm{L2}(\bm{P} \bm{x}) .
$$

L2 normalization and RMSNorm differ only in the radius of the sphere, and both preserve the direction of $\bm{x}$. LayerNorm differs from RMSNorm only by the centering projection $\bm{P}$, which rotates $\bm{x}$ by $\phi = \arctan(\lvert \mu \rvert / \sigma)$. Every other difference comes with this rotation: invariance to shifts along $\bm{1}$, a $(d-2)$-dimensional output set instead of $(d-1)$, and the extra mean statistic LayerNorm has to compute.

## Experiments

### Assumptions
To isolate the effect of each normalization, we compare idealized versions of the three operations:

- **No affine parameters.** We set $\bm{\gamma} = \bm{g} = \bm{1}$ and $\bm{\beta} = \bm{0}$, so any difference in the outputs comes from the normalization itself rather than from learned parameters.
- **Negligible $\epsilon$.** We keep $\epsilon = 10^{-5}$, which is far smaller than the variance of our inputs, so the idealized identities from the previous section hold almost exactly.
- **Synthetic inputs with a controllable mean.** Each input is generated as $\bm{x} = \bm{z} + c\bm{1}$ with $\bm{z} \sim \mathcal{N}(\bm{0}, \bm{I}_d)$. The scalar $c$ controls how much of $\bm{x}$ lies along $\bm{1}$, which is exactly the component that re-centering removes, and therefore how far LayerNorm rotates $\bm{x}$.
- **Dimension.** We use $2{,}000$ inputs with $d = 64$ for quantitative measurements and $100$ inputs with $d = 3$ for visualizing the output geometry.

The scripts that produce the figures and the numbers below are [norm_geometry.py]({{ '/assets/code/normalization/norm_geometry.py' | relative_url }}) and [norm_comparison.py]({{ '/assets/code/normalization/norm_comparison.py' | relative_url }}).

Under these assumptions $\mathrm{RMSNorm}(\bm{x}) = \sqrt{d}\,\mathrm{L2}(\bm{x})$, so RMSNorm behaves like L2 normalization up to a constant scale. The experiments therefore focus on the contrast between L2 normalization (no re-centering, direction preserved) and LayerNorm (re-centering, direction changed).

### Visualization

<div class="post-figure-row">
{% include figure.html src="/assets/images/blog/normalization/norm_geometry_c0.png" caption="$c = 0$: the inputs are centered at the origin. L2 normalization and RMSNorm only rescale each input, mapping it onto the sphere of radius $1$ and $\sqrt{3}$ respectively, so their outputs keep the directions of the inputs. LayerNorm first removes the component along $\bm{1}$, which changes the directions, and its outputs lie on a circle of radius $\sqrt{3}$ in the plane orthogonal to $\bm{1}$." %}
{% include figure.html src="/assets/images/blog/normalization/norm_geometry_c2.png" caption="$c = 2$: the inputs are shifted along $\bm{1}$. L2 normalization and RMSNorm preserve the directions of the inputs, including this shared shift, so their outputs collapse towards the $\bm{1}$ direction. LayerNorm removes the shift, rotating every input, and its outputs are exactly the same points as for $c = 0$, since both panels use the same $\bm{z}$." %}
</div>

## Comparison

We now compare the three operations under the assumptions above: no affine parameters, $\epsilon = 10^{-5}$, and inputs $\bm{x} = \bm{z} + c\bm{1}$ with $\bm{z} \sim \mathcal{N}(\bm{0}, \bm{I}_d)$. The comparison is organized around one question: does the operation preserve the direction of $\bm{x}$, and what follows from the answer? All numbers below come from $2{,}000$ inputs of dimension $d = 64$ and are collected under *Measured values* at the end of this section. Each one can be checked against a closed-form prediction.

### L2 normalization

$$
\mathrm{L2}(\bm{x}) = \frac{\bm{x}}{\lVert \bm{x} \rVert_2} .
$$

L2 normalization only rescales $\bm{x}$ along its own ray. Every output lies on the unit sphere and points exactly in the direction of $\bm{x}$: the measured angle between $\bm{x}$ and $\mathrm{L2}(\bm{x})$ is $0.0^\circ$ for every $c$. Because nothing is subtracted, the shared component $c\bm{1}$ survives normalization and dominates the direction as $c$ grows. For two independent inputs $\bm{x}$ and $\bm{x}'$, we have $\bm{x}^\top \bm{x}' \approx d c^2$ and $\lVert \bm{x} \rVert_2^2 \approx d(1 + c^2)$, so the cosine similarity between their outputs is approximately

$$
\cos\angle(\bm{x}, \bm{x}') \approx \frac{c^2}{1 + c^2} .
$$

When $c = 0$, the outputs are spread over the whole sphere and the mean cosine similarity is $0.00$. When $c = 2$, it is already $0.80$, and at $c = 5$ it reaches $0.96$, matching the prediction $c^2 / (1 + c^2)$ at every $c$. The outputs collapse into a small cap around $\bm{1}$, and different inputs become hard to tell apart.

### RMSNorm

Neglecting $\epsilon$,

$$
\mathrm{RMSNorm}(\bm{x}) = \frac{\bm{x}}{\mathrm{RMS}(\bm{x})} = \sqrt{d} \, \frac{\bm{x}}{\lVert \bm{x} \rVert_2} = \sqrt{d} \; \mathrm{L2}(\bm{x}) .
$$

RMSNorm is L2 normalization followed by the constant scale $\sqrt{d}$. Its outputs lie on the sphere of radius $\sqrt{d}$, on the same rays as the L2 outputs, so it preserves the direction of $\bm{x}$ as well and its cosine similarities are identical to those of L2 normalization. The only discrepancy comes from $\epsilon$: the largest entry-wise gap between $\mathrm{RMSNorm}(\bm{x})$ and $\sqrt{d} \, \mathrm{L2}(\bm{x})$ is $3.0 \times 10^{-5}$ at $c = 0$, and it shrinks as $c$ grows, down to $3.8 \times 10^{-7}$ at $c = 5$, because the relative effect of $\epsilon$ is about $\epsilon / (2 \, \mathrm{RMS}(\bm{x})^2)$ and $\mathrm{RMS}(\bm{x})^2 \approx 1 + c^2$. The choice between the two therefore only fixes the output scale. Every geometric property of L2 normalization carries over, including the fact that the shared component $c\bm{1}$ is kept.

### LayerNorm

As shown in the Mathematical analysis, LayerNorm first removes the mean with the centering projection $\bm{P} = \bm{I} - \frac{1}{d} \bm{1} \bm{1}^\top$ and then applies RMSNorm:

$$
\mathrm{LayerNorm}(\bm{x}) = \frac{\bm{P} \bm{x}}{\sigma} = \sqrt{d} \, \frac{\bm{P} \bm{x}}{\lVert \bm{P} \bm{x} \rVert_2} = \mathrm{RMSNorm}(\bm{P} \bm{x}) .
$$

Unlike the other two, LayerNorm changes the direction of its input. As derived in the Mathematical analysis, the angle $\phi$ between $\bm{x}$ and $\mathrm{LayerNorm}(\bm{x})$ satisfies

$$
\tan\phi = \frac{\lvert \mu \rvert}{\sigma} .
$$

For our inputs $\mu \approx c$ and $\sigma \approx 1$, so $\phi \approx \arctan(c)$ once $c$ is not small. In the experiment, the mean angle grows from $5.8^\circ$ at $c = 0$ to $78.8^\circ$ at $c = 5$, and the measured angle agrees with $\arctan(\lvert \mu \rvert / \sigma)$ for every individual input up to $10^{-10}$ degrees. The $5.8^\circ$ at $c = 0$ is not zero because a finite sample of $d = 64$ entries never has exactly zero mean: $\lvert \mu \rvert / \sigma$ is about $\sqrt{2/\pi} / \sqrt{d} \approx 0.10$, which gives $\arctan(0.10) \approx 5.7^\circ$, close to the measured mean.

This rotation is what makes LayerNorm robust to the shared component. Its outputs satisfy both $\lVert \bm{y} \rVert_2 = \sqrt{d}$ and $\bm{1}^\top \bm{y} = 0$, so they lie on a $(d-2)$-dimensional sphere, which is a circle of radius $\sqrt{3}$ when $d = 3$. Since $\bm{P}(\bm{z} + c\bm{1}) = \bm{P} \bm{z}$, the shift $c\bm{1}$ is removed exactly. In the experiment, the LayerNorm outputs for any $c$ differ from those for $c = 0$ by at most $2 \times 10^{-15}$, which is floating-point round-off, and the mean cosine similarity between different outputs stays at $0.00$ for every $c$.

### Measured values

Each quantity is averaged over the $2{,}000$ inputs, or over all pairs of inputs, at every value of $c$. The rows marked *predicted* are the closed-form approximations derived above.

**Rotation of the input: mean angle between $\bm{x}$ and its normalized output.**

| | $c = 0$ | $c = 0.5$ | $c = 1$ | $c = 2$ | $c = 3$ | $c = 5$ |
|---|---|---|---|---|---|---|
| L2 normalization, RMSNorm | $0.0^\circ$ | $0.0^\circ$ | $0.0^\circ$ | $0.0^\circ$ | $0.0^\circ$ | $0.0^\circ$ |
| LayerNorm | $5.8^\circ$ | $26.6^\circ$ | $45.2^\circ$ | $63.6^\circ$ | $71.7^\circ$ | $78.8^\circ$ |
| LayerNorm, predicted $\arctan c$ | $0.0^\circ$ | $26.6^\circ$ | $45.0^\circ$ | $63.4^\circ$ | $71.6^\circ$ | $78.7^\circ$ |
{: .bordered}

**Collapse of the outputs: mean cosine similarity between the outputs of two different inputs.**

| | $c = 0$ | $c = 0.5$ | $c = 1$ | $c = 2$ | $c = 3$ | $c = 5$ |
|---|---|---|---|---|---|---|
| L2 normalization, RMSNorm | $0.00$ | $0.20$ | $0.50$ | $0.80$ | $0.90$ | $0.96$ |
| L2 normalization, RMSNorm, predicted $c^2 / (1 + c^2)$ | $0.00$ | $0.20$ | $0.50$ | $0.80$ | $0.90$ | $0.96$ |
| LayerNorm | $0.00$ | $0.00$ | $0.00$ | $0.00$ | $0.00$ | $0.00$ |
{: .bordered}

**RMSNorm versus rescaled L2: largest entry-wise difference between $\mathrm{RMSNorm}(\bm{x})$ and $\sqrt{d} \, \mathrm{L2}(\bm{x})$.**

| | $c = 0$ | $c = 0.5$ | $c = 1$ | $c = 2$ | $c = 3$ | $c = 5$ |
|---|---|---|---|---|---|---|
| Measured | $3.0 \times 10^{-5}$ | $2.5 \times 10^{-5}$ | $1.2 \times 10^{-5}$ | $3.5 \times 10^{-6}$ | $1.3 \times 10^{-6}$ | $3.8 \times 10^{-7}$ |
{: .bordered}

### At a glance

| | L2 normalization | RMSNorm | LayerNorm |
|---|---|---|---|
| Output | $\bm{x} / \lVert \bm{x} \rVert_2$ | $\sqrt{d} \, \bm{x} / \lVert \bm{x} \rVert_2$ | $\sqrt{d} \, \bm{P} \bm{x} / \lVert \bm{P} \bm{x} \rVert_2$ |
| Direction of $\bm{x}$ preserved | yes | yes | no, rotated by $\arctan(\lvert \mu \rvert / \sigma)$ |
| Re-centering | no | no | yes |
| Output set | sphere of radius $1$ | sphere of radius $\sqrt{d}$ | $(d-2)$-sphere of radius $\sqrt{d}$ in $\bm{1}^\perp$ |
| Invariant to $\bm{x} \mapsto \alpha \bm{x}$, $\alpha > 0$ | yes | yes | yes |
| Invariant to $\bm{x} \mapsto \bm{x} + c \bm{1}$ | no | no | yes |
| Degrees of freedom removed | 1 (norm) | 1 (norm) | 2 (norm and mean) |
| Mean cosine similarity at $c = 5$ | $0.96$ | $0.96$ | $0.00$ |
{: .bordered}

Three points stand out:

1. **L2 normalization and RMSNorm preserve the direction of $\bm{x}$.** Both only change its length and are the same operation up to a constant scale; in the experiment their outputs agree up to $3 \times 10^{-5}$, the effect of $\epsilon$. Any difference between them in practice comes from the output scale and the learned gain, not from the geometry.
2. **LayerNorm changes the direction of $\bm{x}$, and every other difference comes with that rotation.** Re-centering rotates the input by $\arctan(\lvert \mu \rvert / \sigma)$. In exchange, LayerNorm is invariant to a shared shift, which keeps its outputs from collapsing towards $\bm{1}$ (mean cosine similarity $0.00$ instead of $0.96$ at $c = 5$).
3. **How large the rotation is depends on the input.** For roughly zero-mean inputs in high dimension, $\lvert \mu \rvert / \sigma$ is of order $1 / \sqrt{d}$, so the rotation is small ($5.8^\circ$ at $d = 64$, and smaller for larger $d$) and LayerNorm behaves almost like RMSNorm. The two only diverge when the inputs carry a large component along $\bm{1}$.

## Conclusion

The central difference between L2 normalization and LayerNorm is what happens to the direction of the input. **L2 normalization preserves the direction of $\bm{x}$; LayerNorm changes it.**

L2 normalization only rescales $\bm{x}$ along its own ray. Its output is parallel to the input, so the angle between them is always $0^\circ$, and only the length of $\bm{x}$ is lost. RMSNorm is the same operation with a different radius, $\mathrm{RMSNorm}(\bm{x}) = \sqrt{d} \; \mathrm{L2}(\bm{x})$, so it preserves the direction too; in our experiment the two agree up to $3 \times 10^{-5}$, the effect of $\epsilon$.

LayerNorm first removes the mean, $\bm{x} \mapsto \bm{P} \bm{x}$, and only then normalizes, $\mathrm{LayerNorm}(\bm{x}) = \mathrm{RMSNorm}(\bm{P} \bm{x})$. Removing the component $\mu \bm{1}$ rotates the input, and the angle between $\bm{x}$ and its output is

$$
\phi = \arctan\frac{\lvert \mu \rvert}{\sigma} .
$$

The rotation is set entirely by the ratio of the mean to the standard deviation. In our experiment it grows from $5.8^\circ$ at $c = 0$ to $78.8^\circ$ at $c = 5$, matching this formula for every input.

The other differences between the two follow from this one:

- **Shift invariance.** Because L2 normalization keeps the direction, it also keeps any component shared by all inputs. As that component grows, the outputs collapse towards $\bm{1}$, and the mean cosine similarity between different outputs reaches $0.96$ at $c = 5$. LayerNorm discards exactly that component, so its outputs are invariant to shifts along $\bm{1}$ and stay spread out, with a mean cosine similarity of $0.00$ for every $c$.
- **Output geometry.** L2 normalization keeps every direction and maps onto a full $(d-1)$-dimensional sphere. LayerNorm keeps only directions orthogonal to $\bm{1}$ and maps onto a $(d-2)$-dimensional sphere, removing one extra degree of freedom.

So the choice between them is a trade-off. Preserving the direction keeps all the information in $\bm{x}$, including the shared component. Changing the direction buys robustness to that component, at the price of distorting the input.

How large this distortion is in practice depends on the input. For roughly zero-mean inputs in high dimension, $\lvert \mu \rvert / \sigma$ is of order $1 / \sqrt{d}$. The rotation is then small ($5.8^\circ$ at $d = 64$, and smaller for larger $d$), and LayerNorm behaves almost like RMSNorm, which preserves the direction. In that regime, dropping re-centering costs little and saves a statistic and a bias vector per layer. This is consistent with the hypothesis of Zhang and Sennrich[^rmsnorm] that re-scaling invariance, not re-centering invariance, is what normalization needs, and with the move from LayerNorm in GPT-2[^gpt2] and BERT[^bert] to RMSNorm in LLaMA[^llama].

Our analysis uses an idealized setting: no affine parameters, negligible $\epsilon$ and synthetic Gaussian inputs. In a trained model the learned gain and bias reshape the output sphere into an ellipsoid, and real hidden states are not isotropic Gaussians. A natural next step is to measure $\lvert \mu \rvert / \sigma$ on the hidden states of a real LLM. That ratio fixes the angle by which LayerNorm would rotate them, and it shows how far real hidden states are from the regime where preserving or changing the direction makes no difference.

## References
[^ln]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

[^rmsnorm]: Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

[^llama]: Touvron, H., Lavril, T., Izacard, G., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

[^gpt2]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Technical Report. [Semantic Scholar](https://api.semanticscholar.org/CorpusID:160025533)

[^bert]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

[^transformer]: Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
