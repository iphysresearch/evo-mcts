// Gallery functionality
let currentSlideIndex = 0;
const slides = [
    {
        title: "Framework Architecture",
        description: "Integration of LLM guidance, evolutionary optimization, and Monte Carlo tree search components"
    },
    {
        title: "MCTS Tree Structure", 
        description: "Visualization of the Monte Carlo tree search exploration process and node expansion"
    },
    {
        title: "Algorithm Evolution",
        description: "Performance trajectory showing the evolution of algorithmic solutions over iterations"
    },
    {
        title: "Performance Analysis",
        description: "Comparative analysis of detection performance on MLGWSC-1 benchmark dataset"
    },
    {
        title: "Signal Detection Results",
        description: "Examples of gravitational wave signal detection using discovered algorithms"
    },
    {
        title: "Discovery Process",
        description: "Step-by-step visualization of automated algorithmic discovery workflow"
    }
];

function initializeGallery() {
    const indicatorsContainer = document.getElementById('galleryIndicators');
    const totalSlidesElement = document.getElementById('totalSlides');
    
    if (indicatorsContainer) {
        // Create indicators
        slides.forEach((_, index) => {
            const indicator = document.createElement('div');
            indicator.className = `indicator ${index === 0 ? 'active' : ''}`;
            indicator.onclick = () => goToSlide(index);
            indicatorsContainer.appendChild(indicator);
        });
    }
    
    if (totalSlidesElement) {
        totalSlidesElement.textContent = slides.length;
    }
    
    updateSlideInfo();
}

function updateSlideInfo() {
    const currentSlideElement = document.getElementById('currentSlide');
    const descriptionElement = document.getElementById('galleryDescription');
    const indicators = document.querySelectorAll('.indicator');
    
    if (currentSlideElement) {
        currentSlideElement.textContent = currentSlideIndex + 1;
    }
    
    if (descriptionElement) {
        descriptionElement.textContent = slides[currentSlideIndex].description;
    }
    
    // Update indicators
    indicators.forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentSlideIndex);
    });
    
    // Update navigation buttons
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    
    if (prevBtn) {
        prevBtn.disabled = currentSlideIndex === 0;
    }
    
    if (nextBtn) {
        nextBtn.disabled = currentSlideIndex === slides.length - 1;
    }
}

function goToSlide(index) {
    if (index >= 0 && index < slides.length) {
        currentSlideIndex = index;
        const track = document.getElementById('galleryTrack');
        if (track) {
            track.style.transform = `translateX(-${currentSlideIndex * 100}%)`;
        }
        updateSlideInfo();
    }
}

function nextSlide() {
    if (currentSlideIndex < slides.length - 1) {
        goToSlide(currentSlideIndex + 1);
    }
}

function previousSlide() {
    if (currentSlideIndex > 0) {
        goToSlide(currentSlideIndex - 1);
    }
}

// Keyboard navigation for gallery
function handleGalleryKeyboard(e) {
    if (e.key === 'ArrowLeft') {
        previousSlide();
    } else if (e.key === 'ArrowRight') {
        nextSlide();
    }
}

// Navigation functionality
// Image Modal Functions
function openModal(img) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.backgroundColor = 'white';
    modalImage.src = img.src;
    modalImage.alt = img.alt;
    modalImage.style.maxWidth = '90vw';
    modalImage.style.maxHeight = '90vh';
    modalImage.style.objectFit = 'contain';
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
    
    // Add click event to close modal when clicking anywhere
    modal.addEventListener('click', closeModal);
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
    
    // Restore body scroll
    document.body.style.overflow = 'auto';
    
    // Remove click event listener to prevent memory leaks
    modal.removeEventListener('click', closeModal);
}

// Close modal when clicking outside the image
document.getElementById('imageModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // Initialize gallery
    initializeGallery();
    // Mobile navigation toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    navToggle.addEventListener('click', function() {
        navToggle.classList.toggle('active');
        navMenu.classList.toggle('active');
    });
    
    // Close mobile menu when clicking on a link
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetElement.offsetTop - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Active navigation link highlighting
    function updateActiveNavLink() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
        const navHeight = document.querySelector('.navbar').offsetHeight;
        
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - navHeight - 100;
            const sectionHeight = section.offsetHeight;
            const scrollPosition = window.pageYOffset;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentSection = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    }
    
    // Update active nav link on scroll
    window.addEventListener('scroll', updateActiveNavLink);
    
    // Navbar background on scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    });
    
    // Tab functionality for Quick Start section
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
    
    // Copy code functionality
    window.copyCode = function(button) {
        const codeBlock = button.closest('.code-block');
        const code = codeBlock.querySelector('code');
        const text = code.textContent;
        
        // Create temporary textarea to copy text
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        
        // Visual feedback
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.style.color = '#10b981';
        
        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.style.color = '';
        }, 2000);
    };
    
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.algorithm-card, .result-card, .team-member, .workflow-step');
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Counter animation for stats
    function animateCounters() {
        const counters = document.querySelectorAll('.stat-number, .result-number');
        
        counters.forEach(counter => {
            const target = parseFloat(counter.textContent.replace('%', ''));
            const isPercentage = counter.textContent.includes('%');
            let current = 0;
            const increment = target / 60; // 60 frames for smooth animation
            
            const updateCounter = () => {
                if (current < target) {
                    current += increment;
                    if (current > target) current = target;
                    
                    if (isPercentage) {
                        counter.textContent = current.toFixed(1) + '%';
                    } else {
                        counter.textContent = Math.round(current);
                    }
                    
                    requestAnimationFrame(updateCounter);
                }
            };
            
            // Start animation when element is visible
            const counterObserver = new IntersectionObserver(function(entries) {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        updateCounter();
                        counterObserver.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.5 });
            
            counterObserver.observe(counter);
        });
    }
    
    animateCounters();
    
    // Parallax effect for hero section
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const heroBackground = document.querySelector('.hero-background');
        
        if (heroBackground) {
            const rate = scrolled * -0.5;
            heroBackground.style.transform = `translateY(${rate}px)`;
        }
    });
    
    // Loading animation
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
        
        // Animate hero elements
        const heroElements = document.querySelectorAll('.hero-badges, .hero-title, .hero-description, .hero-stats, .hero-actions');
        heroElements.forEach((el, index) => {
            setTimeout(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, index * 200);
        });
    });
    
    // Initialize hero animations
    const heroElements = document.querySelectorAll('.hero-badges, .hero-title, .hero-description, .hero-stats, .hero-actions');
    heroElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
    });
    
    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        // ESC to close mobile menu
        if (e.key === 'Escape') {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        }
        
        // Gallery keyboard navigation
        handleGalleryKeyboard(e);
    });
    
    // External links handling
    const externalLinks = document.querySelectorAll('a[target="_blank"]');
    externalLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Add analytics or tracking here if needed
            console.log('External link clicked:', this.href);
        });
    });
    
    // Search functionality (if needed in the future)
    function initSearch() {
        // Placeholder for search functionality
        // Can be implemented later if needed
    }
    
    // Performance monitoring
    if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'largest-contentful-paint') {
                    console.log('LCP:', entry.startTime);
                }
            }
        });
        
        observer.observe({ entryTypes: ['largest-contentful-paint'] });
    }
    
    // Service worker registration (for future PWA features)
    if ('serviceWorker' in navigator) {
        // Uncomment when service worker is implemented
        // navigator.serviceWorker.register('/sw.js');
    }
    
    // Dark mode toggle (placeholder for future implementation)
    function toggleDarkMode() {
        // Implementation for dark mode
        // Can be added later based on user preferences
    }
    
    // Initialize MathJax if present
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // Could send to analytics or error reporting service
});

// Accessibility improvements
document.addEventListener('DOMContentLoaded', function() {
    // Add focus indicators for keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });
    
    // Skip to main content link
    const skipLink = document.createElement('a');
    skipLink.href = '#home';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'skip-link';
    skipLink.style.cssText = `
        position: absolute;
        top: -40px;
        left: 6px;
        background: #2563eb;
        color: white;
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        z-index: 1001;
        transition: top 0.3s;
    `;
    
    skipLink.addEventListener('focus', function() {
        this.style.top = '6px';
    });
    
    skipLink.addEventListener('blur', function() {
        this.style.top = '-40px';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);
});

// Analytics placeholder
function trackEvent(action, category, label) {
    // Placeholder for analytics tracking
    // Can be integrated with Google Analytics, Plausible, etc.
    console.log('Event tracked:', { action, category, label });
}

// Export functions for potential use in other scripts
window.EvoMCTS = {
    copyCode: window.copyCode,
    trackEvent: trackEvent,
    debounce: debounce,
    throttle: throttle
};
