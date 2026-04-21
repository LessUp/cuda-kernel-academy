/**
 * Post-build script for GitHub Pages optimization
 * Generates sitemap.xml, robots.txt, and adds SEO meta tags
 */

const fs = require('fs');
const path = require('path');

const SITE_URL = 'https://lessup.github.io/cuda-kernel-academy';
const BUILD_DIR = path.join(__dirname, '..', '_site');

// ===== Generate Sitemap =====
function generateSitemap() {
    const pages = [];
    const htmlFiles = [];

    function findHtmlFiles(dir, basePath = '') {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            const relativePath = path.join(basePath, entry.name);

            if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'gitbook') {
                findHtmlFiles(fullPath, relativePath);
            } else if (entry.isFile() && entry.name.endsWith('.html')) {
                htmlFiles.push(relativePath);
            }
        }
    }

    findHtmlFiles(BUILD_DIR);

    const now = new Date().toISOString();

    let sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
`;

    // Add index page first with highest priority
    sitemap += `  <url>
    <loc>${SITE_URL}/</loc>
    <lastmod>${now}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
`;

    for (const file of htmlFiles) {
        if (file === 'index.html') continue;

        const urlPath = file.replace(/\\/g, '/').replace(/\.html$/, '');
        const fullUrl = urlPath === 'index' ? SITE_URL : `${SITE_URL}/${urlPath}`;

        // Determine priority based on path depth
        const depth = urlPath.split('/').length;
        const priority = Math.max(0.3, 1.0 - (depth - 1) * 0.2).toFixed(1);

        sitemap += `  <url>
    <loc>${fullUrl}</loc>
    <lastmod>${now}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>${priority}</priority>
  </url>
`;
    }

    sitemap += `</urlset>`;

    fs.writeFileSync(path.join(BUILD_DIR, 'sitemap.xml'), sitemap);
    console.log('✓ Generated sitemap.xml');
}

// ===== Generate Robots.txt =====
function generateRobots() {
    const robots = `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml

# Disallow gitbook internal paths
Disallow: /gitbook/
`;

    fs.writeFileSync(path.join(BUILD_DIR, 'robots.txt'), robots);
    console.log('✓ Generated robots.txt');
}

// ===== Add SEO Meta Tags =====
function addSEOMetaTags() {
    const indexPath = path.join(BUILD_DIR, 'index.html');

    if (!fs.existsSync(indexPath)) {
        console.log('⚠ index.html not found, skipping SEO meta tags');
        return;
    }

    let content = fs.readFileSync(indexPath, 'utf8');

    // Check if already has meta tags
    if (content.includes('og:title')) {
        console.log('✓ SEO meta tags already present');
        return;
    }

    const metaTags = `
    <meta name="description" content="CUDA Kernel Academy - Systematic learning path for high-performance CUDA kernel development. From SGEMM basics to Tensor Core mastery. | 从零到极致：系统性学习 CUDA 高性能算子开发">
    <meta name="keywords" content="CUDA, GPU, HPC, SGEMM, GEMM, Tensor Core, 高性能计算, 深度学习, 推理引擎, 矩阵乘法, 并行计算, TensorCraft, FlashAttention">
    <meta name="author" content="LessUp">
    <meta name="robots" content="index, follow">
    <meta name="googlebot" content="index, follow">

    <!-- Open Graph -->
    <meta property="og:title" content="CUDA Kernel Academy - Systematic CUDA High-Performance Kernel Development">
    <meta property="og:description" content="From SGEMM basics to Tensor Core mastery. A comprehensive learning path with 4 progressive modules.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="${SITE_URL}/">
    <meta property="og:image" content="${SITE_URL}/assets/images/og-image.png">
    <meta property="og:locale" content="en_US">
    <meta property="og:locale:alternate" content="zh_CN">
    <meta property="og:site_name" content="CUDA Kernel Academy">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="CUDA Kernel Academy - High-Performance CUDA Kernel Development">
    <meta name="twitter:description" content="From SGEMM basics to Tensor Core mastery. 4 progressive modules for systematic learning.">
    <meta name="twitter:image" content="${SITE_URL}/assets/images/og-image.png">
    <meta name="twitter:creator" content="@LessUp">

    <!-- Canonical URL -->
    <link rel="canonical" href="${SITE_URL}/">

    <!-- Favicon -->
    <link rel="icon" type="image/svg+xml" href="/assets/images/favicon.svg">
    <link rel="alternate icon" type="image/x-icon" href="/assets/images/favicon.ico">
    <link rel="apple-touch-icon" sizes="180x180" href="/assets/images/apple-touch-icon.png">

    <!-- Structured Data -->
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": "CUDA Kernel Academy",
        "description": "Systematic learning path for high-performance CUDA kernel development - from SGEMM basics to Tensor Core mastery",
        "author": {
            "@type": "Organization",
            "name": "LessUp"
        },
        "publisher": {
            "@type": "Organization",
            "name": "LessUp",
            "url": "https://github.com/LessUp"
        },
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": "${SITE_URL}/"
        },
        "isAccessibleForFree": true,
        "educationalLevel": "intermediate",
        "learningResourceType": "tutorial",
        "inLanguage": ["en", "zh"],
        "about": {
            "@type": "Thing",
            "name": "CUDA Programming",
            "sameAs": "https://en.wikipedia.org/wiki/CUDA"
        }
    }
    </script>
`;

    // Insert after <head>
    content = content.replace('<head>', `<head>${metaTags}`);

    // Add custom CSS link
    const cssLink = `    <link rel="stylesheet" href="/assets/styles/website.css">\n`;
    content = content.replace('</head>', `${cssLink}</head>`);

    fs.writeFileSync(indexPath, content);
    console.log('✓ Added SEO meta tags to index.html');

    // Process all other HTML files
    function processHtmlFiles(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);

            if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'gitbook') {
                processHtmlFiles(fullPath);
            } else if (entry.isFile() && entry.name.endsWith('.html')) {
                if (fullPath === indexPath) continue;

                let fileContent = fs.readFileSync(fullPath, 'utf8');

                // Extract title from h1
                const titleMatch = fileContent.match(/<h1[^>]*>(.*?)<\/h1>/);
                const title = titleMatch ? titleMatch[1].replace(/<[^>]*>/g, '') : 'CUDA Kernel Academy';

                // Basic meta tags for other pages
                const basicMeta = `
    <meta name="robots" content="index, follow">
    <link rel="stylesheet" href="/assets/styles/website.css">
`;

                if (!fileContent.includes('website.css')) {
                    fileContent = fileContent.replace('</head>', `${basicMeta}</head>`);
                    fs.writeFileSync(fullPath, fileContent);
                }
            }
        }
    }

    processHtmlFiles(BUILD_DIR);
    console.log('✓ Added CSS to all HTML files');
}

// ===== Generate 404 Page =====
function generate404() {
    const notFoundHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found - CUDA Kernel Academy</title>
    <meta name="description" content="Page not found">
    <meta name="robots" content="noindex">
    <link rel="stylesheet" href="/assets/styles/website.css">
    <style>
        :root {
            --color-brand: #76B900;
            --color-brand-dark: #5a8f00;
            --color-text: #2c3e50;
            --color-text-light: #5d6d7e;
            --color-bg: #ffffff;
            --shadow-md: 0 4px 6px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.08);
            --radius-md: 8px;
        }
        body {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            background: var(--color-bg);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
        }
        .error-container {
            text-align: center;
            padding: 2em;
            max-width: 600px;
        }
        .error-code {
            font-size: 8em;
            font-weight: 700;
            color: var(--color-brand);
            line-height: 1;
            margin-bottom: 0.2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .error-title {
            font-size: 1.8em;
            color: var(--color-text);
            margin-bottom: 0.5em;
        }
        .error-subtitle {
            font-size: 1.2em;
            color: var(--color-text);
            margin-bottom: 1em;
        }
        .error-message {
            color: var(--color-text-light);
            margin-bottom: 2em;
            line-height: 1.6;
        }
        .home-link {
            display: inline-block;
            padding: 0.8em 2em;
            background: var(--color-brand);
            color: white;
            text-decoration: none;
            border-radius: var(--radius-md);
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-md);
        }
        .home-link:hover {
            background: var(--color-brand-dark);
            transform: translateY(-2px);
        }
        .search-box {
            margin: 2em 0;
        }
        .search-box input {
            width: 100%;
            max-width: 400px;
            padding: 0.8em 1em;
            border: 2px solid #e1e8ed;
            border-radius: var(--radius-md);
            font-size: 1em;
            transition: border-color 0.2s ease;
        }
        .search-box input:focus {
            outline: none;
            border-color: var(--color-brand);
        }
        .links {
            margin-top: 2em;
        }
        .links a {
            color: var(--color-brand);
            text-decoration: none;
            margin: 0 1em;
        }
        .links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-code">404</div>
        <h1 class="error-title">Page Not Found</h1>
        <h2 class="error-subtitle">页面未找到</h2>
        <p class="error-message">
            The page you're looking for doesn't exist or has been moved.<br>
            抱歉，您访问的页面不存在或已被移动。
        </p>
        <div class="search-box">
            <input type="text" placeholder="Search documentation..." id="searchInput" onkeydown="if(event.key==='Enter') search()">
        </div>
        <a href="/" class="home-link">Return Home / 返回首页</a>
        <div class="links">
            <a href="https://github.com/LessUp/cuda-kernel-academy">GitHub</a>
            <a href="https://github.com/LessUp/cuda-kernel-academy/issues">Report Issue</a>
        </div>
    </div>
    <script>
        function search() {
            const query = document.getElementById('searchInput').value;
            if (query) {
                window.location.href = '/?q=' + encodeURIComponent(query);
            }
        }
    </script>
</body>
</html>`;

    fs.writeFileSync(path.join(BUILD_DIR, '404.html'), notFoundHtml);
    console.log('✓ Generated 404.html');
}

// ===== Main =====
function main() {
    console.log('🔧 Post-build optimization starting...\n');

    if (!fs.existsSync(BUILD_DIR)) {
        console.error('❌ Build directory not found:', BUILD_DIR);
        process.exit(1);
    }

    try {
        generateSitemap();
        generateRobots();
        addSEOMetaTags();
        generate404();

        console.log('\n✅ Post-build optimization completed!');
    } catch (error) {
        console.error('\n❌ Error during post-build:', error.message);
        process.exit(1);
    }
}

main();
