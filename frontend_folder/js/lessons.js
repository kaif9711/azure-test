// Lessons loader. Fetches lessons/<lang>.json and exposes window.loadLessons()
(function(){
  async function fetchLessons(lang){
    const res = await fetch(`lessons/${lang}.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error('Failed to load lessons');
    return res.json();
  }

  async function loadLessons(containerSelector = '#lessons-container'){
    const lang = (window.I18n && I18n.current) || 'en';
    try {
      const data = await fetchLessons(lang);
      const container = document.querySelector(containerSelector);
      if (!container) return data;
      container.innerHTML = '';
      data.lessons.forEach(lsn => {
        const card = document.createElement('div');
        card.className = 'card mb-3';
        card.innerHTML = `
          <div class="card-body">
            <h5 class="card-title">${lsn.title}</h5>
            <p class="card-text">${lsn.instructions}</p>
            ${Array.isArray(lsn.exercises) && lsn.exercises.length ? `
              <ul class="mb-0">
                ${lsn.exercises.map(it => `<li>${it}</li>`).join('')}
              </ul>` : ''}
          </div>
        `;
        container.appendChild(card);
      });
      return data;
    } catch (err) {
      console.error('[lessons] load failed:', err);
      return null;
    }
  }

  window.loadLessons = loadLessons;
})();
