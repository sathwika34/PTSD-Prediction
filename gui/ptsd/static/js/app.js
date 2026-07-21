/**
 * PTSD Risk Detection System — App JavaScript
 * Clinical Precision Framework
 */

// ── Scroll Animations (IntersectionObserver) ──────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Observe elements with .animate-in for scroll-triggered animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -40px 0px'
    });

    document.querySelectorAll('.animate-in').forEach(el => {
        observer.observe(el);
    });

    // ── Counter Animation for Metric Cards ──
    document.querySelectorAll('.metric-card-value').forEach(el => {
        const text = el.textContent.trim();
        const match = text.match(/^([\d.]+)(%?)$/);
        if (match) {
            const target = parseFloat(match[1]);
            const suffix = match[2] || '';
            animateCounter(el, target, suffix);
        }
    });

    // ── Active nav highlighting based on scroll ──
    highlightActiveNav();
});

/**
 * Animate a counter from 0 to target value
 */
function animateCounter(element, target, suffix = '', duration = 1200) {
    const startTime = performance.now();
    const isDecimal = target % 1 !== 0;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;

        if (isDecimal) {
            element.textContent = current.toFixed(1) + suffix;
        } else {
            element.textContent = Math.round(current) + suffix;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Highlight active navigation link
 */
function highlightActiveNav() {
    const path = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === path) {
            link.classList.add('active');
        }
    });
}

/**
 * Smooth page transitions
 */
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        const main = document.querySelector('.main-content');
        main.style.opacity = '0.6';
        main.style.transition = 'opacity 150ms ease';
        
        setTimeout(() => {
            main.style.opacity = '1';
        }, 200);
    });
});

/**
 * Chart.js Global Defaults
 */
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#43474e';
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    
    // Custom plugin for rounded bars
    Chart.defaults.plugins.tooltip.backgroundColor = '#022448';
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.titleFont = { family: 'Inter', size: 12, weight: '600' };
    Chart.defaults.plugins.tooltip.bodyFont = { family: 'Inter', size: 11 };
}
