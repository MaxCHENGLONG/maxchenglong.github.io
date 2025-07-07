---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I hold a Master's degree in Machine Learning and Statistics from the Department of Information Technology, Uppsala University, and will commence my PhD studies at Linköping University in September 2025.

I will soon begin my PhD at Linköping University, focusing on machine learning, particularly generative models and instabilities arising from cycles of retraining on mixtures of human- and machine-generated data. My research aims to better understand model collapse and contribute to the development of more stable and robust AI systems.

## Publications

{% for post in site.publications reversed %}
  - **{{ post.title }}**<br>
    {{ post.authors }}<br>
    <em>{{ post.journal }}</em>, {{ post.date | date: "%Y" }}<br>
    [Link]({{ post.url | relative_url }})
{% endfor %}