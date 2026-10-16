---
published: false  # example post, not built; copy this file as a starting point, or delete this line to publish it
title: "Example Post"
description: "Text, math, images and interactive figures in one post."
scripts:
  - https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js
---

This is an example blog post. Write normal **Markdown** here: paragraphs, lists, links, code and tables all work.

## Math

Inline math uses double dollars, like $$\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$$. A display equation goes in its own paragraph:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

## Images

{% include figure.html src="/assets/images/covers/cover1.jpg" caption="An image with a caption. Put the images for each article in <code>assets/images/blog/&lt;article-name&gt;/</code>." width="70%" %}

## Interactive figure written in the article

The heatmap below is drawn by a script inside this file. Drag the slider to change the softmax temperature.

<div class="post-figure">
<div id="attention-heatmap" style="height: 420px;"></div>
<label for="attention-temperature">Temperature <strong id="attention-temperature-value"></strong></label>
<input id="attention-temperature" type="range" min="0.1" max="5" step="0.1" value="1" style="width: 60%; vertical-align: middle;">
<figcaption>Attention weights of each query token over all key tokens.</figcaption>
</div>

{% raw %}
<script>
(function () {
  var tokens = ["The", "cat", "sat", "on", "the", "mat"];
  var scores = tokens.map(function (_, i) {
    return tokens.map(function (_, j) { return 2 * Math.cos(0.9 * i - 0.7 * j) + (i === j ? 1 : 0); });
  });
  var slider = document.getElementById("attention-temperature");
  var label = document.getElementById("attention-temperature-value");
  function softmaxRows(T) {
    return scores.map(function (row) {
      var e = row.map(function (s) { return Math.exp(s / T); });
      var sum = e.reduce(function (a, b) { return a + b; }, 0);
      return e.map(function (v) { return v / sum; });
    });
  }
  function draw() {
    var T = parseFloat(slider.value);
    label.textContent = T.toFixed(1);
    Plotly.react("attention-heatmap", [{
      z: softmaxRows(T), x: tokens, y: tokens, type: "heatmap",
      colorscale: "Blues", reversescale: true, zmin: 0, zmax: 1
    }], {
      margin: { t: 10, r: 10, b: 50, l: 60 },
      xaxis: { title: { text: "key" } },
      yaxis: { title: { text: "query" }, autorange: "reversed" }
    }, { responsive: true, displayModeBar: false });
  }
  slider.addEventListener("input", draw);
  draw();
})();
</script>
{% endraw %}

## Interactive figure from a standalone page

Bigger demos can live in their own folder under `demos/` and be embedded here. This one is `demos/softmax-temperature/index.html`.

{% include interactive.html src="/demos/softmax-temperature/" height="420px" caption="Softmax over one query's scores." %}

## Code

```python
import torch
weights = torch.softmax(scores / temperature, dim=-1)
```
