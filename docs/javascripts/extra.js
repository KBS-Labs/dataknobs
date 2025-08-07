// Custom JavaScript for Dataknobs documentation

// Add copy button to code blocks (if not already provided by theme)
document.addEventListener('DOMContentLoaded', function() {
    // Custom initialization code here
    console.log('Dataknobs documentation loaded');
});

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});