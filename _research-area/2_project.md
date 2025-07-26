---
layout: page
title: Whole Slide Image (WSI) foundation model
description: Learning-based representation of WSIs
img: assets/img/digital-pathology.jpg
importance: 2
category: Digital pathology
related_publications: true
---

A proper embedding model is the basis for any machine learning approach. There have been many attempts at training a foundation embedding model for WSIs. Most of them rely on the vision transformer architecture, with minor modifications. However, WSIs are different in that the density of information is much lower compared to other images.

I believe in order to develop a truly foundational model for WSIs, a different approach with a dedicated architecture is necessary. On this topic, I intend to tackle some of the problems that modern vision architectures have and to come up with a foundation model for WSIs that overcome such obstacles.