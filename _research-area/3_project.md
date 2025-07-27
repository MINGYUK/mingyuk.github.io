---
layout: page
title: Weighting algorithms for causal inference
description: Weighting algorithms such as IPTW, employed for estimation of treatment effects.
img: assets/img/causal-inference.png
importance: 1
category: Causal inference
related_publications: false
---

Causal inference is a process of identifying causal relationship. The gold standard for establishing causality is randomised-controlled trial (RCT). In reality, there are many barriers to conducting one: they are expensive, they take time, and sometimes the question at hand cannot be turned into an RCT.

Causal inference tries to estimate the true causal relationship without conducting a trial. There are multiple methods to achieve this. In medicine, propensity score matching is the most widely accepted. However, this cannot always be done, for example due to small sample size. In such a case, weighting algorithms like inverse probability of treatment weighting (IPTW) can be an option.

In this area, I study algorithms that can be used to estimate treatment effect. This includes testing established algorithms on real data, and confirming their behavior in different distributions or assumptions.
