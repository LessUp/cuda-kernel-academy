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
    <meta name="description" content="从零到极致：系统性学习 CUDA 高性能算子开发。包含 SGEMM 教程、TensorCraft 核心算子库、HPC 进阶优化和推理引擎实现的完整学习路径。">
    <meta name="keywords" content="CUDA, GPU, 高性能计算, HPC, SGEMM, GEMM, TensorCraft, 深度学习, 推理引擎, 矩阵乘法, 并行计算">
    <meta name="author" content="LessUp">
    <meta name="robots" content="index, follow">
    <meta name="googlebot" content="index, follow">

    <!-- Open Graph -->
    <meta property="og:title" content="CUDA Kernel Academy - 系统性学习 CUDA 高性能算子开发">
    <meta property="og:description" content="从零到极致：系统性学习 CUDA 高性能算子开发。包含 SGEMM 教程、TensorCraft 核心算子库、HPC 进阶优化和推理引擎实现。">
    <meta property="og:type" content="website">
    <meta property="og:url" content="${SITE_URL}/">
    <meta property="og:image" content="${SITE_URL}/assets/images/og-image.png">
    <meta property="og:locale" content="zh_CN">
    <meta property="og:site_name" content="CUDA Kernel Academy">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="CUDA Kernel Academy - 系统性学习 CUDA 高性能算子开发">
    <meta name="twitter:description" content="从零到极致：系统性学习 CUDA 高性能算子开发">
    <meta name="twitter:image" content="${SITE_URL}/assets/images/og-image.png">

    <!-- Canonical URL -->
    <link rel="canonical" href="${SITE_URL}/">

    <!-- Favicon -->
    <link rel="icon" type="image/x-icon" href="/assets/images/favicon.ico">
    <link rel="apple-touch-icon" sizes="180x180" href="/assets/images/apple-touch-icon.png">
    <link rel="apple-touch-icon" sizes="152x152" href="/assets/images/apple-touch-icon-152x152.png">

    <!-- Structured Data -->
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": "CUDA Kernel Academy",
        "description": "从零到极致：系统性学习 CUDA 高性能算子开发",
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
        "learningResourceType": "tutorial"
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
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>页面未找到 - CUDA Kernel Academy</title>
    <meta name="robots" content="noindex">
    <link rel="stylesheet" href="/assets/styles/website.css">
    <style>
        body {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            background: var(--color-bg);
        }
        .error-container {
            text-align: center;
            padding: 2em;
            max-width: 600px;
        }
        .error-code {
            font-size: 6em;
            font-weight: 700;
            color: var(--color-brand);
            line-height: 1;
            margin-bottom: 0.2em;
        }
        .error-title {
            font-size: 1.5em;
            color: var(--color-text);
            margin-bottom: 1em;
        }
        .error-message {
            color: var(--color-text-light);
            margin-bottom: 2em;
        }
        .home-link {
            display: inline-block;
            padding: 0.8em 1.5em;
            background: var(--color-brand);
            color: white;
            text-decoration: none;
            border-radius: var(--radius-md);
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .home-link:hover {
            background: var(--color-brand-dark);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .search-box {
            margin: 2em 0;
        }
        .search-box input {
            width: 100%;
            max-width: 400px;
            padding: 0.8em 1em;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            font-size: 1em;
        }
        .search-box input:focus {
            outline: none;
            border-color: var(--color-brand);
            box-shadow: 0 0 0 3px rgba(118, 185, 0, 0.15);
        }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-code">404</div>
        <h1 class="error-title">页面未找到</h1>
        <p class="error-message">抱歉，您访问的页面不存在或已被移动。</p>
        <div class="search-box">
            <input type="text" placeholder="搜索文档..." id="searchInput" onkeydown="if(event.key==='Enter') search()">
        </div>
        <a href="/" class="home-link">返回首页</a>
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
