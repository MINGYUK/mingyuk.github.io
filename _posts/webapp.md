---
layout: post
title: Developing and deploying a web app
date: 2025-08-01 00:01:00
description: Issues I faced servicing a simple web app
tags:
  - web
categories:
  - record-keeping
pseudocode: false
toc:
  sidebar: false
featured: false
tabs: false
pretty_table: false
---
# Intro
I noticed I was slouching too much on my chair, so I decided I must do something about it. I keep my browser open when I work, so I settled on using Javascript to bulid a simple web app that looks at me when I'm in front of my computer and tracks my posture.

JS already had a well-documented module named [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) for pose estimation that runs on edge devices, maintained by Google. The model was only a few MBs, and it was implemented using WebAssembly, which enabled real-time tracking. I was able to quickly write and serve the app on localhost.

# The Problem
I tested the app on localhost using ```python -m http.server 8080```, which did not cause any problem. However when I tried to access the service from another computer using IP:port, I found out that camera permission is not allowed with http. It had to be served with https to be accessible from other machines within the local network.

Python does not have any built-in https module, so I had to write a script using ssl. While I could access the service from other machines after this, the browsers prompted the user for not visiting a safe site that had a certificate not signed by a Certificate Authority (CA).

This was not ideal for me, I intended to actually serve this on the web as a toy project. After some research I found out:
 - I need a domain to get any CA to sign my certificate. Luckily I already had one
 - cloudflare tunnels can support functions similar to nginx
