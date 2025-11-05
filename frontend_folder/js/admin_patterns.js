(function(){
  const apiBase = window.API_BASE || 'http://localhost:8000';
  const msgEl = document.getElementById('message');
  const tableBody = document.querySelector('#patterns-table tbody');
  const loadBtn = document.getElementById('load-btn');
  const showInactiveCb = document.getElementById('show-inactive');
  const createForm = document.getElementById('create-form');
  const editModal = document.getElementById('edit-modal');
  const editForm = document.getElementById('edit-form');
  const editIdSpan = document.getElementById('edit-id');
  const cancelEditBtn = document.getElementById('cancel-edit');

  function jwt() { return document.getElementById('jwt').value.trim(); }
  function authHeaders() { return { 'Authorization': 'Bearer ' + jwt(), 'Content-Type': 'application/json' }; }
  function setMessage(txt, good=false) { msgEl.textContent = txt; msgEl.style.color = good ? 'green' : 'crimson'; }

  async function loadPatterns(){
    if(!jwt()) { setMessage('Provide JWT'); return; }
    tableBody.innerHTML = '<tr><td colspan="7">Loading...</td></tr>';
    try {
      const r = await fetch(`${apiBase}/patterns?include_inactive=${showInactiveCb.checked}`, { headers: authHeaders() });
      if(!r.ok) { throw new Error('Failed: ' + r.status); }
      const data = await r.json();
      renderRows(data);
      setMessage('Loaded ' + data.length + ' patterns', true);
    } catch(e){
      setMessage('Error loading patterns: ' + e.message);
      tableBody.innerHTML = '';
    }
  }

  function renderRows(patterns){
    tableBody.innerHTML = '';
    if(!patterns.length){ tableBody.innerHTML = '<tr><td colspan="7">No patterns</td></tr>'; return; }
    patterns.forEach(p => {
      const tr = document.createElement('tr');
      if(!p.is_active) tr.classList.add('inactive');
      tr.innerHTML = `<td>${p.id}</td><td>${p.pattern_name}</td><td>${p.pattern_type}</td><td>${p.risk_weight.toFixed(2)}</td>` +
                     `<td>${p.is_active ? 'Active' : 'Inactive'}</td><td>${p.updated_at}</td>` +
                     `<td><button data-edit="${p.id}">Edit</button></td>`;
      tableBody.appendChild(tr);
    });
  }

  tableBody.addEventListener('click', (e)=>{
    const btn = e.target.closest('button[data-edit]');
    if(!btn) return;
    const id = btn.getAttribute('data-edit');
    startEdit(id);
  });

  async function startEdit(id){
    // try to fetch row fresh (list already has it, but we keep simple)
    const row = [...tableBody.querySelectorAll('tr')].find(tr => tr.firstChild.textContent == id);
    if(!row){ setMessage('Row not found'); return; }
    editIdSpan.textContent = id;
    editForm.pattern_name.value = row.children[1].textContent;
    editForm.pattern_type.value = row.children[2].textContent;
    editForm.risk_weight.value = row.children[3].textContent;
    editForm.description.value = '';
    editForm.is_active.checked = row.children[4].textContent === 'Active';
    editModal.classList.remove('hidden');
  }

  createForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    if(!jwt()) { setMessage('Provide JWT'); return; }
    const formData = new FormData(createForm);
    const payload = Object.fromEntries(formData.entries());
    payload.risk_weight = parseFloat(payload.risk_weight);
    payload.is_active = formData.get('is_active') === 'on';
    try {
      const r = await fetch(`${apiBase}/patterns`, { method:'POST', headers: authHeaders(), body: JSON.stringify(payload) });
      if(!r.ok) { const t = await r.text(); throw new Error(t); }
      await loadPatterns();
      createForm.reset();
      setMessage('Pattern created', true);
    } catch(err){ setMessage('Create failed: ' + err.message); }
  });

  editForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    if(!jwt()) { setMessage('Provide JWT'); return; }
    const id = editIdSpan.textContent;
    const formData = new FormData(editForm);
    const payload = {};
    for(const [k,v] of formData.entries()){
      if(v) payload[k] = v;
    }
    if(formData.get('risk_weight')) payload.risk_weight = parseFloat(formData.get('risk_weight'));
    payload.is_active = formData.get('is_active') === 'on';
    try {
      const r = await fetch(`${apiBase}/patterns/${id}`, { method:'PUT', headers: authHeaders(), body: JSON.stringify(payload) });
      if(!r.ok) { const t = await r.text(); throw new Error(t); }
      editModal.classList.add('hidden');
      await loadPatterns();
      setMessage('Pattern updated', true);
    } catch(err){ setMessage('Update failed: ' + err.message); }
  });

  cancelEditBtn.addEventListener('click', ()=> editModal.classList.add('hidden'));
  loadBtn.addEventListener('click', loadPatterns);
})();
