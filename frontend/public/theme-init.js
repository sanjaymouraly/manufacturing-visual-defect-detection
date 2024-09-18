// Apply the saved theme before first paint (avoids a flash of the wrong theme).
// Kept as an external same-origin file so the Content-Security-Policy can forbid inline scripts.
try {
  const t = localStorage.getItem("dl-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.classList.toggle("dark", t === "dark");
} catch (e) { /* storage unavailable: keep default theme */ }
