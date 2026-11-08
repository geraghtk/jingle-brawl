// The quick-reference card's Print button (a file, not an inline onclick: the CSP runs only our own scripts).
document.getElementById('print')?.addEventListener('click', () => window.print());
