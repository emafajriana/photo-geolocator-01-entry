# Photo Geolocator v3
Model-free geolocation from JPEG DC thumbnails, using normalized color and road/vegetation cues.
An upside-down skylike test flips the raster before features are measured.
Cue fractions weight nineteen worldwide zones, blended on the sphere using zone priors.
The blended estimate is shrunk 70/30 toward the global prior at (56.07, 9.93) and rounded to six decimals.
The deterministic env driver retains four concurrent photo fetches and checked sequential answer batches of five.
