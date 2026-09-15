/**
 * Inline script that runs before React hydrates and applies the saved
 * theme to <html> so the user never sees a flash of the wrong theme.
 */
const SCRIPT = `
(function () {
  try {
    var saved = window.localStorage.getItem('lstm-theme');
    var theme = saved === 'light' || saved === 'dark'
      ? saved
      : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    var root = document.documentElement;
    if (theme === 'dark') root.classList.add('dark'); else root.classList.remove('dark');
    root.style.colorScheme = theme;
  } catch (e) {
    /* localStorage unavailable; fall back to light */
  }
})();
`;

export function ThemeScript() {
  return <script dangerouslySetInnerHTML={{ __html: SCRIPT }} />;
}
