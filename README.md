# Photo Geolocator
The agent makes no model calls: it parses JPEG Huffman streams and reconstructs an RGB thumbnail from DC coefficients.
It extracts sky, ground, greenery, saturation, snow, aridity, blue-sky, and greyscale features.
Ordered threshold rules classify each scene to one of seven fixed geographic anchors.
A deterministic byte hash adds bounded coordinate jitter, and malformed images use the jittered fallback anchor.
The env driver fetches four photos concurrently and posts checked answers in ordered chunks of five.
Baseline and progressive synthetic JPEG fixtures match downscaled PIL sky/ground means; the 25-photo stub replay matches.
