// Simple i18n utility for frontend UI translations
// Usage: add data-i18n="key.path" to elements. Optional: data-i18n-attr="placeholder|title|aria-label".
// Switch language by calling I18n.setLang('en'|'ar'|...).

(function () {
  const STORAGE_KEY = 'app_lang';

  const isRtl = (lang) => ['ar', 'fa', 'he', 'ur'].includes(lang);

  const I18n = {
    current: 'en',
    translations: {},
    init(defaultLang = 'en') {
      const saved = localStorage.getItem(STORAGE_KEY) || defaultLang;
      this.setLang(saved);
      // Delegate clicks on language switch items
      document.addEventListener('click', (e) => {
        const el = e.target.closest('[data-lang]');
        if (el && el.dataset.lang) {
          e.preventDefault();
          this.setLang(el.dataset.lang);
        }
      });
    },
    async setLang(lang) {
      try {
        // Load translation JSON
        const res = await fetch(`i18n/${lang}.json`, { cache: 'no-cache' });
        if (!res.ok) throw new Error('Failed to load translations');
        this.translations = await res.json();
        this.current = lang;
        localStorage.setItem(STORAGE_KEY, lang);

        // Update html lang and direction
        const html = document.documentElement;
        html.lang = lang;
        html.dir = isRtl(lang) ? 'rtl' : 'ltr';

        // Apply translations to DOM
        this.apply();

        // Update any visible indicator
        const indicator = document.querySelector('[data-i18n-current-lang]');
        if (indicator) indicator.textContent = lang.toUpperCase();
      } catch (err) {
        console.warn('[i18n] Falling back to English due to error:', err.message);
        if (lang !== 'en') {
          this.setLang('en');
        }
      }
    },
    t(key, fallback) {
      const parts = key.split('.');
      let val = this.translations;
      for (const p of parts) {
        if (val && Object.prototype.hasOwnProperty.call(val, p)) {
          val = val[p];
        } else {
          return fallback !== undefined ? fallback : key;
        }
      }
      return typeof val === 'string' ? val : (fallback !== undefined ? fallback : key);
    },
    apply(root = document) {
      // Text content
      root.querySelectorAll('[data-i18n]')?.forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (!key) return;
        const attr = el.getAttribute('data-i18n-attr');
        const value = this.t(key, el.textContent?.trim());
        if (attr) {
          // Support comma-separated list of attributes
          attr.split(',').forEach((a) => el.setAttribute(a.trim(), value));
        } else {
          el.textContent = value;
        }
      });
      // Placeholders
      root.querySelectorAll('[data-i18n-placeholder]')?.forEach((el) => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (key) el.setAttribute('placeholder', this.t(key, el.getAttribute('placeholder') || ''));
      });
      // Titles
      root.querySelectorAll('[data-i18n-title]')?.forEach((el) => {
        const key = el.getAttribute('data-i18n-title');
        if (key) el.setAttribute('title', this.t(key, el.getAttribute('title') || ''));
      });
    },
  };

  // Expose globally
  window.I18n = I18n;
})();
