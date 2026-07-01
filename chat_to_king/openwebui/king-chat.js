(function () {
  function setFavicon() {
    try {
      var links = document.querySelectorAll('link[rel~="icon"]');
      links.forEach(function (l) {
        if (l.href.indexOf('/albedo-favicon.svg') === -1) l.href = '/albedo-favicon.svg';
      });
      if (!links.length) {
        var l = document.createElement('link');
        l.rel = 'icon'; l.type = 'image/svg+xml'; l.href = '/albedo-favicon.svg';
        document.head.appendChild(l);
      }
    } catch (e) {}
  }

  var overlay;
  function ensureOverlay() {
    if (overlay) return overlay;
    var st = document.createElement('style');
    st.textContent = '@keyframes krspin{to{transform:rotate(360deg)}}';
    document.head.appendChild(st);
    overlay = document.createElement('div');
    overlay.id = 'albedo-king-reload';
    overlay.style.cssText =
      'position:fixed;inset:0;z-index:2147483647;display:none;flex-direction:column;' +
      'align-items:center;justify-content:center;gap:18px;background:#0b0b0d;color:#e8e8ea;' +
      'font-family:Inter,system-ui,-apple-system,sans-serif;text-align:center;padding:24px';
    overlay.innerHTML =
      '<img src="/albedo-favicon.svg" width="76" height="76" alt="Albedo">' +
      '<div style="font-size:20px;font-weight:600;letter-spacing:.2px">A new king is being crowned</div>' +
      '<div style="font-size:14px;opacity:.65;max-width:380px;line-height:1.5">' +
      'Loading the new model onto the GPUs — this usually takes a minute or two. ' +
      'The chat will resume automatically.</div>' +
      '<div style="margin-top:6px;width:26px;height:26px;border:3px solid #2a2a2e;' +
      'border-top-color:#e8e8ea;border-radius:50%;animation:krspin 1s linear infinite"></div>';
    document.body.appendChild(overlay);
    return overlay;
  }

  function poll() {
    fetch('/__king/status', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (s) {
        var loading = !!(s && (s.reloading || s.serving === false));
        ensureOverlay().style.display = loading ? 'flex' : 'none';
      })
      .catch(function () {});
  }

  function init() {
    setFavicon();
    poll();
    setInterval(poll, 4000);
    setInterval(setFavicon, 5000);
  }
  if (document.body) init();
  else document.addEventListener('DOMContentLoaded', init);
})();
