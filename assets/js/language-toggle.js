/**
 * Language Toggle for SGEMM Optimization Docs
 * Handles switching between English and Chinese pages
 * Auto-redirects based on browser language preference
 */

(function() {
  'use strict';

  // Page pair mappings - maps page_key to language-specific paths
  const pagePairs = {
    'home': { en: '/', 'zh-CN': '/zh/' },
    'specs': { en: '/specs/', 'zh-CN': '/zh/specs/' },
    'getting-started': { en: '/docs/getting-started/', 'zh-CN': '/zh/docs/getting-started/' },
    'learning-path': { en: '/docs/learning-path/', 'zh-CN': '/zh/docs/learning-path/' },
    'architecture': { en: '/docs/architecture/', 'zh-CN': '/zh/docs/architecture/' },
    'benchmark-results': { en: '/docs/benchmark-results/', 'zh-CN': '/zh/docs/benchmark-results/' },
    'kernel-naive': { en: '/docs/kernel-naive/', 'zh-CN': '/zh/docs/kernel-naive/' },
    'kernel-tiled': { en: '/docs/kernel-tiled/', 'zh-CN': '/zh/docs/kernel-tiled/' },
    'kernel-bank-free': { en: '/docs/kernel-bank-free/', 'zh-CN': '/zh/docs/kernel-bank-free/' },
    'kernel-double-buffer': { en: '/docs/kernel-double-buffer/', 'zh-CN': '/zh/docs/kernel-double-buffer/' },
    'kernel-tensor-core': { en: '/docs/kernel-tensor-core/', 'zh-CN': '/zh/docs/kernel-tensor-core/' },
    // Chinese page keys (for reverse lookup)
    'zh-home': { en: '/', 'zh-CN': '/zh/' },
    'zh-specs': { en: '/specs/', 'zh-CN': '/zh/specs/' },
    'zh-getting-started': { en: '/docs/getting-started/', 'zh-CN': '/zh/docs/getting-started/' },
    'zh-learning-path': { en: '/docs/learning-path/', 'zh-CN': '/zh/docs/learning-path/' },
    'zh-architecture': { en: '/docs/architecture/', 'zh-CN': '/zh/docs/architecture/' },
    'zh-benchmark-results': { en: '/docs/benchmark-results/', 'zh-CN': '/zh/docs/benchmark-results/' },
    'zh-kernel-naive': { en: '/docs/kernel-naive/', 'zh-CN': '/zh/docs/kernel-naive/' },
    'zh-kernel-tiled': { en: '/docs/kernel-tiled/', 'zh-CN': '/zh/docs/kernel-tiled/' },
    'zh-kernel-bank-free': { en: '/docs/kernel-bank-free/', 'zh-CN': '/zh/docs/kernel-bank-free/' },
    'zh-kernel-double-buffer': { en: '/docs/kernel-double-buffer/', 'zh-CN': '/zh/docs/kernel-double-buffer/' },
    'zh-kernel-tensor-core': { en: '/docs/kernel-tensor-core/', 'zh-CN': '/zh/docs/kernel-tensor-core/' }
  };

  // Key for storing redirect state
  const REDIRECT_KEY = 'sgemm-lang-redirected';

  // Get base URL from Jekyll site
  const baseurl = document.querySelector('meta[name="baseurl"]')?.content ||
                  document.documentElement.getAttribute('data-baseurl') ||
                  '/sgemm-optimization';

  /**
   * Detect browser language preference
   * Returns 'zh-CN' for Chinese browsers, 'en' otherwise
   */
  function detectBrowserLanguage() {
    const browserLang = navigator.language || navigator.userLanguage || 'en';
    // Match zh-CN, zh-TW, zh-HK, zh-SG, or plain zh
    return /^zh/i.test(browserLang) ? 'zh-CN' : 'en';
  }

  /**
   * Check if this session has already handled language redirect
   */
  function hasRedirectedThisSession() {
    try {
      return sessionStorage.getItem(REDIRECT_KEY) === 'true';
    } catch (e) {
      return false;
    }
  }

  /**
   * Mark that redirect has been handled this session
   */
  function markRedirectHandled() {
    try {
      sessionStorage.setItem(REDIRECT_KEY, 'true');
    } catch (e) {
      // sessionStorage not available
    }
  }

  /**
   * Check if user has explicitly set language preference
   */
  function hasExplicitPreference() {
    try {
      return localStorage.getItem('preferred-doc-lang') !== null;
    } catch (e) {
      return false;
    }
  }

  /**
   * Auto-redirect based on browser language (first visit only)
   */
  function autoRedirect() {
    // Skip if already redirected this session or user has explicit preference
    if (hasRedirectedThisSession() || hasExplicitPreference()) {
      return;
    }

    // Mark that we've handled redirect for this session
    markRedirectHandled();

    const browserLang = detectBrowserLanguage();
    const currentPath = window.location.pathname;

    // Check if we're on an English page (not /zh/)
    const isEnglishPage = !currentPath.includes('/zh/');
    const isHomePage = currentPath === baseurl + '/' || currentPath === baseurl;

    // Only redirect from English home page to Chinese if browser is Chinese
    if (isHomePage && browserLang === 'zh-CN') {
      window.location.replace(baseurl + '/zh/');
    }
  }

  /**
   * Store language preference
   */
  function setLanguagePreference(lang) {
    try {
      localStorage.setItem('preferred-doc-lang', lang);
    } catch (e) {
      // localStorage not available
    }
  }

  /**
   * Get stored language preference
   */
  function getLanguagePreference() {
    try {
      return localStorage.getItem('preferred-doc-lang');
    } catch (e) {
      return null;
    }
  }

  /**
   * Find the paired page path for a given page key and target language
   */
  function findPairedPath(pageKey, targetLang) {
    const pair = pagePairs[pageKey];
    if (pair && pair[targetLang]) {
      return baseurl + pair[targetLang];
    }
    // Fallback to language home
    return targetLang === 'zh-CN' ? baseurl + '/zh/' : baseurl + '/';
  }

  /**
   * Initialize language switcher functionality
   */
  function initSwitcher() {
    const switcher = document.querySelector('.language-switcher');
    if (!switcher) return;

    const pageKey = switcher.dataset.pageKey;
    const currentLang = switcher.dataset.lang;

    // Add click handlers to buttons
    const buttons = switcher.querySelectorAll('button[data-lang-choice]');
    buttons.forEach(button => {
      button.addEventListener('click', function(e) {
        e.preventDefault();
        const targetLang = this.dataset.langChoice;

        // Skip if already on this language
        if (targetLang === currentLang) return;

        // Store preference
        setLanguagePreference(targetLang);

        // Navigate to paired page
        const targetPath = findPairedPath(pageKey, targetLang);
        window.location.assign(targetPath);
      });
    });
  }

  /**
   * Initialize all language features
   */
  function init() {
    // Run auto-redirect first (only on first visit)
    autoRedirect();

    // Initialize switcher buttons
    initSwitcher();
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
