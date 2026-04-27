/**
 * Language Toggle for SGEMM Optimization Docs
 * Handles switching between English and Chinese pages
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

  // Get base URL from Jekyll site
  const baseurl = document.querySelector('meta[name="baseurl"]')?.content ||
                  document.documentElement.getAttribute('data-baseurl') ||
                  '/sgemm-optimization';

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
   * Initialize language switcher functionality
   */
  function init() {
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

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
