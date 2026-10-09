'use strict';
// Manuscript Table 1. Each row stores [PSNR, SSIM, LPIPS] at x1, x2, x4.
const metrics = [
  [[26.049,.862,.137],[25.782,.868,.207],[24.920,.877,.269]],
  [[26.359,.868,.129],[20.408,.809,.290],[17.966,.755,.425]],
  [[27.504,.890,.112],[20.031,.774,.341],[16.385,.635,.491]],
  [[27.537,.892,.112],[26.331,.866,.217],[24.897,.842,.320]],
  [[27.086,.884,.123],[26.066,.866,.205],[25.118,.857,.276]],
  [[27.523,.891,.118],[26.332,.870,.202],[25.240,.858,.276]],
  [[27.567,.892,.117],[26.357,.871,.202],[25.252,.858,.276]]
];
const captions = ['256 × 256 · Standard rendering','512 × 512 · HRRC ×2','1024 × 1024 · HRRC ×4'];
const notes = [
  'At native resolution, CompactSplat-L is close to SurfSplat in PSNR (+0.030 dB), ties its SSIM, and has higher LPIPS (0.117 vs. 0.112), with 44.6% fewer Gaussians.',
  'At ×2 rendering resolution, CompactSplat-L records 26.357 dB PSNR and 0.202 LPIPS, compared with SurfSplat’s 26.331 dB and 0.217. The number of Gaussians is unchanged.',
  'At ×4 rendering resolution, CompactSplat-L records 25.252 dB PSNR and 0.276 LPIPS, compared with SurfSplat’s 24.897 dB and 0.320, while using 72.6k rather than 131.1k Gaussians.'
];
function showResolution(scale) {
  const rows = document.querySelectorAll('#results-body tr');
  const best = [0,1,2].map(column => (column === 2 ? Math.min : Math.max)(...metrics.map(row => row[scale][column])));
  rows.forEach((row, i) => {
    const cells = row.querySelectorAll('td');
    metrics[i][scale].forEach((value, j) => {
      cells[j].textContent = value.toFixed(3);
      cells[j].classList.toggle('best', value === best[j]);
    });
  });
  document.getElementById('table-caption').textContent = captions[scale];
  document.getElementById('comparison-note').textContent = notes[scale];
}
document.querySelectorAll('input[name="scale"]').forEach(input => {
  input.addEventListener('change', () => showResolution(Number(input.value)));
});
showResolution(0);
document.getElementById('copy-citation').addEventListener('click', async () => {
  const citation = document.getElementById('citation');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
