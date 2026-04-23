const fs = require('fs');
const path = require('path');

const SITE_URL = 'https://lessup.github.io/cuda-kernel-academy';
const SITE_NAME = 'CUDA Kernel Academy';
const SITE_DESCRIPTION =
    'Showcase-first documentation for a multi-module CUDA learning repository covering SGEMM, reusable kernels, advanced optimization, and lightweight inference.';
const BUILD_DIR = path.join(__dirname, '..', '_site');
const INJECT_START = '<!-- cuda-kernel-academy:meta:start -->';
const INJECT_END = '<!-- cuda-kernel-academy:meta:end -->';

function collectHtmlFiles(dir, basePath = '') {
    const files = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.join(basePath, entry.name);

        if (entry.isDirectory()) {
            if (entry.name.startsWith('.') || entry.name === 'gitbook') {
                continue;
            }
            files.push(...collectHtmlFiles(fullPath, relativePath));
            continue;
        }

        if (entry.isFile() && entry.name.endsWith('.html')) {
            files.push(relativePath);
        }
    }

    return files.sort();
}

function toUrlPath(relativeHtmlPath) {
    const normalized = relativeHtmlPath.replace(/\\/g, '/');
    if (normalized === 'index.html') {
        return '';
    }
    if (normalized.endsWith('/index.html')) {
        return normalized.replace(/\/index\.html$/, '');
    }
    return normalized.replace(/\.html$/, '');
}

function toPageUrl(relativeHtmlPath) {
    const urlPath = toUrlPath(relativeHtmlPath);
    return urlPath ? `${SITE_URL}/${urlPath}` : `${SITE_URL}/`;
}

function relativeAssetPath(relativeHtmlPath, assetPath) {
    const normalized = relativeHtmlPath.replace(/\\/g, '/');
    const depth = normalized.split('/').length - 1;
    return `${'../'.repeat(depth)}${assetPath}`;
}

function stripInjectedMeta(content) {
    return content.replace(new RegExp(`${INJECT_START}[\\s\\S]*?${INJECT_END}\\n?`, 'g'), '');
}

function extractTitle(content) {
    const titleMatch = content.match(/<title>(.*?)<\/title>/i);
    if (titleMatch) {
        return titleMatch[1].replace(/\s+/g, ' ').trim();
    }

    const h1Match = content.match(/<h1[^>]*>(.*?)<\/h1>/i);
    if (h1Match) {
        return h1Match[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
    }

    return SITE_NAME;
}

function extractDescription(content) {
    const paragraphMatch = content.match(/<p>(.*?)<\/p>/i);
    if (!paragraphMatch) {
        return SITE_DESCRIPTION;
    }

    const text = paragraphMatch[1]
        .replace(/<[^>]+>/g, '')
        .replace(/\s+/g, ' ')
        .trim();

    return text ? text.slice(0, 180) : SITE_DESCRIPTION;
}

function escapeHtmlAttribute(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function injectMeta(relativeHtmlPath) {
    const fullPath = path.join(BUILD_DIR, relativeHtmlPath);
    const raw = fs.readFileSync(fullPath, 'utf8');
    const content = stripInjectedMeta(raw);
    const title = escapeHtmlAttribute(extractTitle(content));
    const description = escapeHtmlAttribute(extractDescription(content));
    const canonical = toPageUrl(relativeHtmlPath);
    const stylesheet = relativeAssetPath(relativeHtmlPath, 'assets/styles/website.css');
    const favicon = relativeAssetPath(relativeHtmlPath, 'assets/images/favicon.svg');

    const injected = `
${INJECT_START}
    <meta name="description" content="${description}">
    <meta name="robots" content="index, follow">
    <meta name="theme-color" content="#76B900">
    <link rel="canonical" href="${canonical}">
    <meta property="og:title" content="${title}">
    <meta property="og:description" content="${description}">
    <meta property="og:type" content="website">
    <meta property="og:url" content="${canonical}">
    <meta property="og:site_name" content="${SITE_NAME}">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="${title}">
    <meta name="twitter:description" content="${description}">
    <link rel="icon" type="image/svg+xml" href="${favicon}">
    <link rel="stylesheet" href="${stylesheet}">
${INJECT_END}
`;

    fs.writeFileSync(fullPath, content.replace('</head>', `${injected}\n</head>`));
}

function generateSitemap(htmlFiles) {
    const timestamp = new Date().toISOString();
    const urls = htmlFiles
        .map((file) => {
            const urlPath = toUrlPath(file);
            const depth = urlPath ? urlPath.split('/').length : 0;
            const priority = Math.max(0.3, 1.0 - depth * 0.15).toFixed(1);
            return `  <url>
    <loc>${toPageUrl(file)}</loc>
    <lastmod>${timestamp}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>${priority}</priority>
  </url>`;
        })
        .join('\n');

    const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;

    fs.writeFileSync(path.join(BUILD_DIR, 'sitemap.xml'), sitemap);
}

function generateRobots() {
    const robots = `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml

Disallow: /gitbook/
`;
    fs.writeFileSync(path.join(BUILD_DIR, 'robots.txt'), robots);
}

function generate404() {
    const notFoundPage = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found - ${SITE_NAME}</title>
    <meta name="description" content="Page not found">
    <meta name="robots" content="noindex">
    <link rel="icon" type="image/svg+xml" href="assets/images/favicon.svg">
    <link rel="stylesheet" href="assets/styles/website.css">
</head>
<body class="not-found-page">
    <main class="not-found-card">
        <p class="not-found-code">404</p>
        <h1>Page not found</h1>
        <p>The page you requested does not exist or has moved.</p>
        <p><a href="${SITE_URL}/">Back to CUDA Kernel Academy</a></p>
    </main>
</body>
</html>
`;

    fs.writeFileSync(path.join(BUILD_DIR, '404.html'), notFoundPage);
}

function main() {
    if (!fs.existsSync(BUILD_DIR)) {
        throw new Error(`Build directory not found: ${BUILD_DIR}`);
    }

    const htmlFiles = collectHtmlFiles(BUILD_DIR);
    htmlFiles.forEach(injectMeta);
    generateSitemap(htmlFiles);
    generateRobots();
    generate404();

    console.log(`Processed ${htmlFiles.length} HTML files.`);
}

main();
