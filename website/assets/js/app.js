/* Nav scroll state, scroll reveal, lightbox, tab toggle */

(function () {
  'use strict';

  /* ─── Nav scroll state ───────────────────────────────────────── */
  const nav = document.querySelector('.nav');
  function onScroll() {
    if (window.scrollY > 8) nav.classList.add('is-scrolled');
    else nav.classList.remove('is-scrolled');
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ─── Scroll reveal (mark sections / cards) ──────────────────── */
  function tagReveal() {
    const targets = document.querySelectorAll(
      '.hero__inner > *, .section__inner > *, .card, .pipeline__step, .tile, .resource, .limit'
    );
    targets.forEach(el => el.setAttribute('data-reveal', ''));
  }
  tagReveal();

  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('is-in');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('[data-reveal]').forEach(el => io.observe(el));

  /* ─── Tab toggle for results section ─────────────────────────── */
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('is-active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('is-active');
      tab.setAttribute('aria-selected', 'true');
      const scheme = parseInt(tab.dataset.scheme, 10);
      if (window.DGA_CHARTS) window.DGA_CHARTS.renderMethods(scheme);
    });
  });

  /* ─── Lightbox ───────────────────────────────────────────────── */
  const lightbox = document.getElementById('lightbox');
  const lbImg = lightbox.querySelector('.lightbox__img');
  const lbCap = lightbox.querySelector('.lightbox__caption');
  const lbClose = lightbox.querySelector('.lightbox__close');

  document.querySelectorAll('[data-lightbox] img').forEach(img => {
    img.addEventListener('click', () => {
      lbImg.src = img.src;
      lbImg.alt = img.alt;
      const caption = img.closest('figure').querySelector('figcaption');
      lbCap.textContent = caption ? caption.textContent : '';
      lightbox.classList.add('is-open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    });
  });

  function closeLightbox() {
    lightbox.classList.remove('is-open');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  }
  lbClose.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('is-open')) closeLightbox();
  });

  /* ─── Render charts when Plotly is ready ─────────────────────── */
  function tryRender() {
    if (window.Plotly && window.DGA_DATA && window.DGA_CHARTS) {
      window.DGA_CHARTS.renderAll();
    } else {
      setTimeout(tryRender, 80);
    }
  }
  tryRender();

  /* ─── Smooth-scroll offset for sticky nav ────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      if (!id) return;
      const el = document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      const navH = nav.offsetHeight + 8;
      const top = el.getBoundingClientRect().top + window.scrollY - navH;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });
})();
