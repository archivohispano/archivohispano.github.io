// Theme Toggle
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update button icons
    document.querySelectorAll('.theme-toggle, .control-btn, .mobile-text-controls button').forEach(btn => {
        if (btn.textContent === '🌙' || btn.textContent === '☀️') {
            btn.textContent = newTheme === 'dark' ? '☀️' : '🌙';
        }
    });
}

// Language Toggle
function toggleLanguage() {
    const currentPath = window.location.pathname;
    let newPath = currentPath;

    // Remove trailing slash if present (except for root)
    if (currentPath.endsWith('/') && currentPath !== '/') {
        newPath = currentPath.slice(0, -1);
    }

    // Check if we're on a text page with an embedded alt-lang URL
    const textPageData = document.getElementById('text-page-data');
    if (textPageData && textPageData.dataset.altLangUrl) {
        window.location.href = textPageData.dataset.altLangUrl;
        return;
    }

    // Static page mappings (only pages that can't be auto-derived)
    const staticMappings = {
        '/': '/en/',
        '/en': '/',
        '/es/autores': '/en/autores',
        '/en/autores': '/es/autores',
        '/es/paises': '/en/paises',
        '/en/paises': '/es/paises',
        '/es/paises/puerto-rico': '/en/paises/puerto-rico',
        '/en/paises/puerto-rico': '/es/paises/puerto-rico'
    };

    if (staticMappings[newPath]) {
        window.location.href = staticMappings[newPath];
        return;
    }

    // Auto-derive author page toggle: /es/autores/X <-> /en/autores/X
    const authorMatch = newPath.match(/^\/(es|en)\/autores\/(.+)$/);
    if (authorMatch) {
        const targetLang = authorMatch[1] === 'es' ? 'en' : 'es';
        window.location.href = '/' + targetLang + '/autores/' + authorMatch[2];
        return;
    }

    alert('Translation not available for this page / Traducción no disponible para esta página');
}

// Font size control (for text pages)
let currentFontSize = 1.1;
function changeFontSize(delta) {
    const textContent = document.querySelector('.text-content');
    if (textContent) {
        currentFontSize += delta * 0.1;
        currentFontSize = Math.max(0.8, Math.min(1.5, currentFontSize));
        textContent.style.fontSize = currentFontSize + 'rem';
    }
}

// Copy link
function copyLink() {
    navigator.clipboard.writeText(window.location.href);
    alert('Enlace copiado al portapapeles / Link copied to clipboard');
}

// Download text
function downloadText() {
    const textContent = document.querySelector('.text-content');
    const title = document.querySelector('.text-title');
    const author = document.querySelector('.text-author');

    if (textContent && title && author) {
        const text = textContent.innerText;
        const metadata = `${title.innerText}\n${author.innerText}\n\n`;
        const blob = new Blob([metadata + text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${title.innerText.toLowerCase().replace(/ /g, '-')}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Search functionality using Jekyll-generated index
let searchIndex = null;

function loadSearchIndex() {
    if (searchIndex) return Promise.resolve(searchIndex);
    return fetch('/search.json')
        .then(r => r.json())
        .then(data => { searchIndex = data; return data; });
}

function performSearch(query) {
    const q = query.toLowerCase().trim();
    if (!q) return [];

    return searchIndex.filter(item => {
        const fields = [
            item.title,
            item.author,
            item.source,
            item.country,
            (item.collections || []).join(' ')
        ].join(' ').toLowerCase();
        return fields.includes(q);
    });
}

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.querySelectorAll('.theme-toggle, .control-btn, .mobile-text-controls button').forEach(btn => {
            if (btn.textContent === '🌙') btn.textContent = '☀️';
        });
    }

    // Search functionality
    const searchBox = document.querySelector('#searchBox');
    if (searchBox) {
        // Detect language from page
        const pageLang = document.documentElement.lang || 'es';
        searchBox.setAttribute('placeholder',
            pageLang === 'en'
                ? 'Search authors, texts, or themes... (press Enter)'
                : 'Buscar autores, textos, o temas... (presione Enter)'
        );

        searchBox.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const query = e.target.value.trim();
                if (!query) {
                    alert(pageLang === 'en'
                        ? 'Please enter a search term'
                        : 'Por favor ingrese un término de búsqueda');
                    return;
                }

                loadSearchIndex().then(() => {
                    const results = performSearch(query);
                    // Prefer results in the current page language
                    const langResults = results.filter(r => r.lang === pageLang);
                    const best = langResults.length > 0 ? langResults[0] : results[0];

                    if (best) {
                        window.location.href = best.url;
                    } else {
                        alert(pageLang === 'en'
                            ? 'No results found for: ' + query
                            : 'No se encontraron resultados para: ' + query);
                    }
                });
            }
        });
    }
});
