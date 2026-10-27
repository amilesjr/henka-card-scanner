# Henka Crop

Square listing photos from trading card shots, made on your phone:
<https://amilesjr.github.io/henka-card-scanner/>

Choose photos from your camera roll. Each card is found, straightened and centred on a
2000 × 2000 photo with the table colour around it, ready for eBay or a shop. Then save them back to
Photos through the share sheet.

- **Photos never leave the phone.** Everything runs in the browser; nothing is uploaded and nothing is
  stored. Closing the page clears the list.
- Originals are never changed. Only the card is straightened and cropped: nothing on it is cleaned
  up, retouched or redrawn, so wear stays visible.
- Photos are listed, and saved, in the order they were taken. Each saved photo carries only the time
  its original was taken (no location, no camera details), so Photos files it right next to the
  original.
- When the card's outline is uncertain the photo is marked **Check** instead of being saved as if it
  were right. You can keep it, or place the four corners by hand.

This repository holds only the built site. The source is `site/` in the Henka Card Photos project
(`node site/build.js OUT`). Fonts: Shippori Mincho and Zen Kaku Gothic New, under the SIL Open Font
License (`fonts/OFL-*.txt`).
