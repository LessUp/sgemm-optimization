/**
 * Language toggle and auto-redirect for SGEMM Optimization docs.
 * - Keeps paired EN/ZH page switching consistent.
 * - Redirects homepage based on saved preference or browser language.
 */

(function () {
  'use strict';

  const PAGE_PAIRS = {
    home: { en: '/', 'zh-CN': '/zh/' },
    specs: { en: '/specs/', 'zh-CN': '/zh/specs/' },
    'getting-started': { en: '/docs/getting-started/', 'zh-CN': '/zh/docs/getting-started/' },
    'learning-path': { en: '/docs/learning-path/', 'zh-CN': '/zh/docs/learning-path/' },
    architecture: { en: '/docs/architecture/', 'zh-CN': '/zh/docs/architecture/' },
    'benchmark-results': { en: '/docs/benchmark-results/', 'zh-CN': '/zh/docs/benchmark-results/' },
    'kernel-naive': { en: '/docs/kernel-naive/', 'zh-CN': '/zh/docs/kernel-naive/' },
    'kernel-tiled': { en: '/docs/kernel-tiled/', 'zh-CN': '/zh/docs/kernel-tiled/' },
    'kernel-bank-free': { en: '/docs/kernel-bank-free/', 'zh-CN': '/zh/docs/kernel-bank-free/' },
    'kernel-double-buffer': { en: '/docs/kernel-double-buffer/', 'zh-CN': '/zh/docs/kernel-double-buffer/' },
    'kernel-tensor-core': { en: '/docs/kernel-tensor-core/', 'zh-CN': '/zh/docs/kernel-tensor-core/' },
    'optimization-playbook': {
      en: '/docs/optimization-playbook/',
      'zh-CN': '/zh/docs/optimization-playbook/'
    },
    'cuda-memory-cheatsheet': {
      en: '/docs/cuda-memory-cheatsheet/',
      'zh-CN': '/zh/docs/cuda-memory-cheatsheet/'
    },
    'performance-casebook': {
      en: '/docs/performance-casebook/',
      'zh-CN': '/zh/docs/performance-casebook/'
    }
  };

  const REDIRECT_KEY = 'sgemm-lang-redirected';
  const PREFERENCE_KEY = 'preferred-doc-lang';

  function normalizeBase(rawBase) {
    if (!rawBase || rawBase === '/') {
      return '';
    }
    return rawBase.endsWith('/') ? rawBase.slice(0, -1) : rawBase;
  }

  const rawBaseurl = document.querySelector('meta[name="baseurl"]')?.content || '';
  const baseurl = normalizeBase(rawBaseurl);

  function withBase(path) {
    const target = `${baseurl}${path}`;
    return target || '/';
  }

  function normalizePath(path) {
    if (!path) {
      return '/';
    }
    return path.endsWith('/') ? path : `${path}/`;
  }

  function pathEquals(currentPath, targetPath) {
    return normalizePath(currentPath) === normalizePath(withBase(targetPath));
  }

  function detectBrowserLanguage() {
    const browserLang = navigator.language || navigator.userLanguage || 'en';
    return /^zh/i.test(browserLang) ? 'zh-CN' : 'en';
  }

  function getLanguagePreference() {
    try {
      const value = localStorage.getItem(PREFERENCE_KEY);
      return value === 'zh-CN' || value === 'en' ? value : null;
    } catch (_error) {
      return null;
    }
  }

  function setLanguagePreference(lang) {
    try {
      localStorage.setItem(PREFERENCE_KEY, lang);
    } catch (_error) {
      // localStorage may be disabled in private mode.
    }
  }

  function hasRedirectedThisSession() {
    try {
      return sessionStorage.getItem(REDIRECT_KEY) === 'true';
    } catch (_error) {
      return false;
    }
  }

  function markRedirectHandled() {
    try {
      sessionStorage.setItem(REDIRECT_KEY, 'true');
    } catch (_error) {
      // sessionStorage may be unavailable.
    }
  }

  function normalizePageKey(pageKey) {
    if (!pageKey) {
      return '';
    }
    return pageKey.startsWith('zh-') ? pageKey.slice(3) : pageKey;
  }

  function findPairedPath(pageKey, targetLang) {
    const normalizedKey = normalizePageKey(pageKey);
    const pair = PAGE_PAIRS[normalizedKey];

    if (pair && pair[targetLang]) {
      return withBase(pair[targetLang]);
    }

    return targetLang === 'zh-CN' ? withBase('/zh/') : withBase('/');
  }

  function autoRedirect() {
    const currentPath = window.location.pathname;
    const onEnglishHome = pathEquals(currentPath, '/');
    const onChineseHome = pathEquals(currentPath, '/zh/');

    if (!onEnglishHome && !onChineseHome) {
      return;
    }

    const preferredLang = getLanguagePreference();
    if (preferredLang === 'zh-CN' && onEnglishHome) {
      window.location.replace(withBase('/zh/'));
      return;
    }
    if (preferredLang === 'en' && onChineseHome) {
      window.location.replace(withBase('/'));
      return;
    }

    if (hasRedirectedThisSession()) {
      return;
    }
    markRedirectHandled();

    if (detectBrowserLanguage() === 'zh-CN' && onEnglishHome) {
      window.location.replace(withBase('/zh/'));
    }
  }

  function initSwitcher() {
    const switchers = document.querySelectorAll('.language-switcher');
    if (!switchers.length) {
      return;
    }

    switchers.forEach((switcher) => {
      const pageKey = switcher.dataset.pageKey || '';
      const currentLang = switcher.dataset.lang === 'zh-CN' ? 'zh-CN' : 'en';
      const buttons = switcher.querySelectorAll('button[data-lang-choice]');

      buttons.forEach((button) => {
        button.addEventListener('click', (event) => {
          event.preventDefault();

          const targetLang = button.dataset.langChoice;
          if (targetLang !== 'en' && targetLang !== 'zh-CN') {
            return;
          }
          if (targetLang === currentLang) {
            return;
          }

          setLanguagePreference(targetLang);

          const targetPath = findPairedPath(pageKey, targetLang);
          if (normalizePath(targetPath) !== normalizePath(window.location.pathname)) {
            window.location.assign(targetPath);
          }
        });
      });
    });
  }

  function init() {
    autoRedirect();
    initSwitcher();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
