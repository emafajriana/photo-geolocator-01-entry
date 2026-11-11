# Photo Geolocator
Each photo gets a structured first-pass vision estimate, with four requests in flight and results kept in photo order.
Low-confidence or unparsed estimates get one sequential second look focused on text, signs, plates, and road furniture.
Nearby estimates are combined by spherical mean; distant estimates use higher confidence, with ties favoring refinement.
Final coordinates are rounded to six decimals and posted in checked sequential batches of five; missing estimates use (20, 78).
The local process grade completed but was voided before scoring because its unconfigured gateway returned upstream 502 errors.
A local 401 stub smoke test posted all 25 fallback answers twice; both runs matched across 50 model requests and all answers.
