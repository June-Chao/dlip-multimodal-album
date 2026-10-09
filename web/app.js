"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {view: "all", album: null, orientation: "", mode: "hybrid", q: "", sort: "newest", page: 1, items: [], total: 0, stats: {}, albums: [], system: null, detail: null, loading: false, upload: false, request: 0, similar: false};
const iconPaths = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  heart: '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>', search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  folder: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
  shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8 12 3 3 5-6"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>', refresh: '<path d="M20 7v5h-5M4 17v-5h5M5.1 7a8 8 0 0 1 13.8-1L20 8M4 16l1.1 2A8 8 0 0 0 19 17"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>', sort: '<path d="M4 6h16M4 12h11M4 18h6"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>', upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
  sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4"/>'
};
function icon(name) { const span = document.createElement("span"); span.className = "icon"; span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${iconPaths[name] || iconPaths.grid}</svg>`; return span; }
$$('[data-icon]').forEach((el) => el.replaceChildren(icon(el.dataset.icon)));
function element(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; }
function button(text, className, action) { const el = element("button", className, text); el.type = "button"; el.addEventListener("click", action); return el; }
function bytes(n = 0) { if (n < 1024) return `${n} B`; const unit = n < 1048576 ? "KB" : n < 1073741824 ? "MB" : "GB"; const divider = {KB: 1024, MB: 1048576, GB: 1073741824}[unit]; return `${(n / divider).toFixed(1)} ${unit}`; }
function toast(message, error = false) { const el = element("div", `toast${error ? " error" : ""}`, message); $('#toast-region').append(el); setTimeout(() => el.remove(), 5500); }
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: options.body instanceof FormData ? options.headers : {"Content-Type": "application/json", ...options.headers}});
  let data; try { data = await response.json(); } catch { throw new Error("服务返回异常，请刷新页面后重试"); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join("；") : "请求未完成，请重试");
  return data;
}
const json = (method, body) => ({method, body: JSON.stringify(body)});
async function busy(control, action) { if (control.disabled) return; control.disabled = true; try { await action(); } catch (error) { toast(error.message, true); } finally { control.disabled = control.id === "build-index" && state.system?.job.status === "running"; } }
function notice(message = "") { $('#notice').textContent = message; $('#notice').hidden = !message; }
function history() { try { const data = JSON.parse(localStorage.getItem("luma-history") || "[]"); return Array.isArray(data) ? data.filter((x) => typeof x === "string").slice(0, 6) : []; } catch { return []; } }
function renderHistory() {
  const values = history(), root = $('#search-history'); root.replaceChildren(); root.hidden = !values.length;
  if (!values.length) return;
  root.append(element("span", "muted", "最近搜索"));
  values.forEach((term) => root.append(button(term, "history-chip", () => { $('#search-input').value = term; $('#search-form').requestSubmit(); })));
  root.append(button("清空", "history-clear", () => { try { localStorage.removeItem("luma-history"); } catch {} renderHistory(); }));
}
function remember(term) { try { localStorage.setItem("luma-history", JSON.stringify([term, ...history().filter((x) => x !== term)].slice(0, 6))); } catch {} renderHistory(); }
function renderNavigation() {
  $('#count-all').textContent = state.stats.total || 0; $('#count-favorites').textContent = state.stats.favorites || 0;
  $('#storage-info').textContent = `${state.stats.total || 0} 张照片 · ${bytes(state.stats.storage_bytes)}`;
  $$('[data-view]').forEach((el) => el.classList.toggle("active", el.dataset.view === state.view));
  const root = $('#album-list'); root.replaceChildren(); $('#no-albums').hidden = !!state.albums.length;
  state.albums.forEach((album) => {
    const item = button("", `nav-item${state.album === album.id ? " active" : ""}`, () => navigate("album", album.id));
    item.append(icon("folder"), element("span", "", album.name), element("b", "", album.count)); root.append(item);
  });
  const album = state.albums.find((item) => item.id === state.album);
  $('#page-title').replaceChildren(document.createTextNode(state.view === "favorites" ? "我的收藏" : album ? album.name : "全部照片"), element("span", "heading-dot", "."));
  $('#page-description').textContent = album?.description || (state.view === "favorites" ? "把偏爱的瞬间，留在离你最近的地方。" : "让每一帧记忆，都能被轻松找到。");
}
async function refreshInfo() { const [stats, albums] = await Promise.all([api("/api/stats"), api("/api/albums")]); state.stats = stats; state.albums = albums.items; renderNavigation(); }
function filters() { return {favorite: state.view === "favorites", album_id: state.album, orientation: state.orientation || null}; }
function renderGrid() {
  const root = $('#photo-grid'); root.replaceChildren();
  state.items.forEach((photo) => {
    const card = element("article", "photo-card");
    const preview = button("", "photo-preview", () => showPhoto(photo.id)); preview.setAttribute("aria-label", `查看 ${photo.name}`);
    const img = element("img"); img.src = photo.thumbnail_url; img.alt = photo.caption || photo.name; img.loading = "lazy"; preview.append(img);
    const favorite = button("", `favorite-button${photo.favorite ? " is-favorite" : ""}`, () => busy(favorite, async () => {
      const updated = await api(`/api/photos/${photo.id}`, json("PATCH", {favorite: !photo.favorite}));
      Object.assign(photo, updated); await refreshInfo(); if (state.view === "favorites") await loadPhotos(); else renderGrid();
    }));
    favorite.append(icon("heart")); favorite.setAttribute("aria-label", `${photo.favorite ? "取消收藏" : "收藏"} ${photo.name}`); favorite.setAttribute("aria-pressed", String(photo.favorite));
    const info = element("div", "photo-info"); info.append(element("h3", "", photo.name));
    info.append(element("p", "", photo.match_reason ? `${photo.match_reason} · 排序分 ${Number(photo.score).toFixed(3)}` : `${photo.width} × ${photo.height} · ${photo.tags.slice(0, 2).join(" / ") || photo.format}`));
    card.append(preview, favorite, info); root.append(card);
  });
  const empty = !state.items.length; $('#empty-state').hidden = !empty;
  const initial = !state.stats.total && !state.q && !state.album && state.view === "all";
  $('#empty-title').textContent = initial ? "美好的记忆，值得被找到。" : state.q || state.similar ? "还没有找到匹配的照片" : state.album ? "这个相册，等待你的故事" : state.view === "favorites" ? "从一张喜欢的照片开始" : "这个画幅还没有照片";
  $('#empty-description').textContent = initial ? "把照片放进光屿，用一句话找回某个瞬间。所有照片与索引，都安静地留在你的电脑里。" : state.q ? "试试更短的关键词，或为照片添加描述与标签。" : state.album ? "在全部照片中打开详情，勾选相册即可收录。" : "导入照片，或调整筛选条件，继续探索你的影像。";
  $('#empty-import').hidden = !initial; $('#empty-hint').hidden = !initial; $('#feature-row').hidden = !initial; $('#empty-kicker').hidden = !initial;
  $('#result-count').textContent = `${state.total} 张${state.q || state.similar ? "匹配" : ""}照片`;
  $('#load-more').hidden = !!state.q || state.similar || state.items.length >= state.total;
  $('#sort-select').disabled = !!state.q || state.similar;
}
async function loadPhotos(append = false) {
  const request = ++state.request; state.loading = true; $('#loading').hidden = false; $('#load-more').disabled = true;
  if (!append) { state.page = 1; notice(); }
  try {
    let result;
    if (state.q) {
      result = await api("/api/search", json("POST", {q: state.q, k: 200, mode: state.mode, ...filters()}));
    } else {
      const query = new URLSearchParams({page: String(state.page), page_size: "36", sort: state.sort});
      Object.entries(filters()).forEach(([key, value]) => { if (value) query.set(key, value); });
      result = await api(`/api/library?${query}`);
    }
    if (request !== state.request) return;
    state.items = append ? [...state.items, ...result.items] : result.items; state.total = result.total; state.similar = false;
    let message = result.warning || "";
    if (state.q && result.total > result.items.length) message += ` 当前展示前 ${result.items.length} 张，请细化搜索条件。`;
    notice(message); renderGrid();
    $('#search-tip').textContent = state.q ? `${{keyword: "关键词", hybrid: "混合", semantic: "语义"}[result.mode]}检索 · ${result.elapsed_ms} ms` : "用自然语言，找回画面里的记忆";
  } catch (error) { if (request !== state.request) return; if (!append) { state.items = []; state.total = 0; renderGrid(); } else { state.page--; } notice(error.message); }
  finally { if (request === state.request) { state.loading = false; $('#loading').hidden = true; $('#load-more').disabled = false; } }
}
function navigate(view, album = null) { state.view = view; state.album = album; state.similar = false; renderNavigation(); loadPhotos(); }
function showDialog(id) { const dialog = $(id); if (!dialog.open) dialog.showModal(); }
async function showPhoto(id) {
  try { const photo = await api(`/api/photos/${id}`); state.detail = photo; renderDetail(); showDialog('#photo-dialog'); } catch (error) { toast(error.message, true); }
}
function renderDetail(preserveDrafts = false) {
  const photo = state.detail; if (!photo) return;
  const captionDraft = $('#detail-caption').value, tagsDraft = $('#detail-tags').value;
  $('#detail-img').src = photo.url; $('#detail-img').alt = photo.caption || photo.name; $('#detail-name').textContent = photo.name;
  $('#detail-meta').textContent = `${photo.width} × ${photo.height} · ${photo.format} · ${bytes(photo.bytes)}\n导入于 ${new Date(photo.created_at).toLocaleString("zh-CN")}`;
  $('#detail-caption').value = preserveDrafts ? captionDraft : photo.caption; $('#detail-tags').value = preserveDrafts ? tagsDraft : photo.tags.join(", ");
  $('#detail-favorite').replaceChildren(icon("heart"), document.createTextNode(photo.favorite ? "已收藏" : "收藏"));
  $('#detail-favorite').classList.toggle("is-favorite", photo.favorite); $('#detail-favorite').setAttribute("aria-pressed", String(photo.favorite));
  $('#detail-original').href = photo.url;
  const root = $('#detail-albums'); root.replaceChildren(); $('#detail-no-albums').hidden = !!state.albums.length;
  state.albums.forEach((album) => {
    const label = element("label", "album-checkbox"), input = element("input"); input.type = "checkbox"; input.checked = photo.album_ids.includes(album.id);
    input.addEventListener("change", () => busy(input, async () => {
      const adding = input.checked;
      try { await api(`/api/albums/${album.id}/photos/${photo.id}`, {method: adding ? "PUT" : "DELETE"}); state.detail = await api(`/api/photos/${photo.id}`); await refreshInfo(); await loadPhotos(); }
      catch (error) { input.checked = !adding; throw error; }
    })); label.append(input, document.createTextNode(album.name)); root.append(label);
  });
}
async function upload(files) {
  if (state.upload || !files.length) return;
  showDialog('#upload-dialog'); const status = $('#upload-status'); status.replaceChildren();
  if (files.length > 50) { status.append(element("p", "error-text", "每次最多导入 50 张照片，请分批导入。")); return; }
  state.upload = true; $('#upload-zone').disabled = true;
  const form = new FormData(); files.forEach((file) => form.append("files", file)); status.append(element("p", "", `正在导入 ${files.length} 张照片…`));
  try {
    const result = await api("/api/import", {method: "POST", body: form}); status.replaceChildren(element("p", "upload-summary", `导入 ${result.imported.length} 张 · 重复跳过 ${result.duplicates.length} 张 · 失败 ${result.errors.length} 张`));
    if (result.duplicates.length) status.append(element("p", "muted", `重复文件：${result.duplicates.join("、")}`));
    result.errors.forEach((item) => status.append(element("p", "error-text", `${item.name}：${item.error}`)));
    if (result.imported.length) status.append(element("p", "muted", "照片已保存。更新索引后，新照片即可参与语义检索。"));
    await refreshInfo(); await loadPhotos();
  } catch (error) { status.replaceChildren(element("p", "error-text", error.message)); }
  finally { state.upload = false; $('#upload-zone').disabled = false; $('#file-input').value = ""; }
}
let pollTimer;
async function refreshSystem() {
  const system = await api("/api/system"); state.system = system;
  const available = Object.entries(system.models).filter(([, value]) => value.available).map(([key]) => key);
  const running = system.job.status === "running"; $('#engine-label').textContent = running ? "索引构建中" : available.length ? "本地引擎已就绪" : "基础图库已就绪";
  $('#engine-dot').classList.toggle("ready", !!available.length);
  const root = $('#system-models'); root.replaceChildren();
  Object.entries(system.models).forEach(([language, model]) => {
    const row = element("label", "model-row"), check = element("input"); check.type = "checkbox"; check.value = language; check.name = "index-language"; check.checked = model.available; check.disabled = !model.available || running;
    const text = element("div"); text.append(element("strong", "", language === "zh" ? "Chinese-CLIP · 中文语义" : "CLIP · 英文与图片"), element("small", "", `${model.available ? "模型已安装" : "模型待安装"} · 已索引 ${system.indexes[language].count} 张`));
    row.append(check, text); root.append(row);
  });
  const progress = $('#index-progress'); progress.replaceChildren(element("p", "", system.job.message || "尚未建立索引"));
  if (running) { const bar = element("progress"); bar.max = system.job.total || 1; bar.value = system.job.processed || 0; progress.append(bar, element("small", "", `${system.job.processed || 0} / ${system.job.total || 0}`)); }
  if (system.job.error) progress.append(element("p", "error-text", system.job.error));
  $('#system-device').textContent = `运算设备：${system.device}`; $('#build-index').disabled = running || !available.length;
  clearTimeout(pollTimer); if (running) pollTimer = setTimeout(() => refreshSystem().catch((error) => toast(error.message, true)), 1400);
}
$$('[data-close]').forEach((control) => control.addEventListener("click", () => control.closest("dialog").close()));
$$('dialog').forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } }));
$$('[data-view]').forEach((control) => control.addEventListener("click", () => navigate(control.dataset.view)));
$$('[data-orientation]').forEach((control) => control.addEventListener("click", () => { state.orientation = control.dataset.orientation; $$('[data-orientation]').forEach((el) => el.classList.toggle("active", el === control)); loadPhotos(); }));
$$('[data-mode]').forEach((control) => control.addEventListener("click", () => { state.mode = control.dataset.mode; $$('[data-mode]').forEach((el) => el.classList.toggle("selected", el === control)); if (state.q) loadPhotos(); }));
$('#search-form').addEventListener("submit", (event) => { event.preventDefault(); state.q = $('#search-input').value.trim(); if (state.q) remember(state.q); loadPhotos(); });
$('#search-input').addEventListener("search", () => { if (!$('#search-input').value) { state.q = ""; loadPhotos(); } });
$('#sort-select').addEventListener("change", (event) => { state.sort = event.target.value; loadPhotos(); });
$('#load-more').addEventListener("click", () => { if (!state.loading) { state.page++; loadPhotos(true); } });
$('#refresh-library').addEventListener("click", (event) => busy(event.currentTarget, async () => { const result = await api("/api/refresh", {method: "POST"}); await refreshInfo(); await loadPhotos(); toast(result.errors.length ? `图库已更新，${result.errors.length} 个文件无法读取` : "图库已更新"); }));
['#import-button', '#empty-import'].forEach((selector) => $(selector).addEventListener("click", () => showDialog('#upload-dialog')));
$('#upload-zone').addEventListener("click", () => $('#file-input').click());
$('#file-input').addEventListener("change", (event) => upload([...event.target.files]));
let dragDepth = 0;
document.addEventListener("dragenter", (event) => { if ([...event.dataTransfer.types].includes("Files")) { event.preventDefault(); dragDepth++; $('#drop-overlay').hidden = false; } });
document.addEventListener("dragover", (event) => { if ([...event.dataTransfer.types].includes("Files")) event.preventDefault(); });
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $('#drop-overlay').hidden = true; } });
document.addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $('#drop-overlay').hidden = true; upload([...event.dataTransfer.files]); });
$('#new-album').addEventListener("click", () => { $('#album-form').reset(); showDialog('#album-dialog'); });
$('#album-form').addEventListener("submit", (event) => { event.preventDefault(); busy(event.submitter, async () => {
  const album = await api("/api/albums", json("POST", {name: $('#album-name').value.trim(), description: $('#album-description').value.trim()}));
  $('#album-dialog').close(); await refreshInfo(); navigate("album", album.id); toast("相册已创建，可在照片详情中添加照片");
}); });
$('#metadata-form').addEventListener("submit", (event) => { event.preventDefault(); busy(event.submitter, async () => {
  const tags = $('#detail-tags').value.split(/[,，、]/).map((tag) => tag.trim()).filter(Boolean);
  state.detail = await api(`/api/photos/${state.detail.id}`, json("PATCH", {caption: $('#detail-caption').value, tags})); renderDetail(); await loadPhotos(); toast("照片信息已保存");
}); });
$('#detail-favorite').addEventListener("click", (event) => busy(event.currentTarget, async () => { state.detail = await api(`/api/photos/${state.detail.id}`, json("PATCH", {favorite: !state.detail.favorite})); renderDetail(true); await refreshInfo(); await loadPhotos(); }));
$('#auto-tag').addEventListener("click", (event) => busy(event.currentTarget, async () => { state.detail = await api(`/api/photos/${state.detail.id}/tags`, {method: "POST"}); renderDetail(); await loadPhotos(); toast("标签已生成并保存"); }));
$('#detail-similar').addEventListener("click", (event) => busy(event.currentTarget, async () => {
  const result = await api(`/api/photos/${state.detail.id}/similar`, json("POST", {k: 60})); ++state.request; state.q = ""; $('#search-input').value = ""; state.similar = true; state.items = result.items; state.total = result.total;
  $('#photo-dialog').close(); notice(`与「${state.detail.name}」相似的照片 · ${result.elapsed_ms} ms`); renderGrid();
}));
$('#open-system').addEventListener("click", () => { showDialog('#system-dialog'); refreshSystem().catch((error) => toast(error.message, true)); });
$('#refresh-system').addEventListener("click", (event) => busy(event.currentTarget, refreshSystem));
$('#build-index').addEventListener("click", (event) => busy(event.currentTarget, async () => {
  const languages = $$('input[name="index-language"]:checked').map((input) => input.value); if (!languages.length) throw new Error("请选择至少一种已安装的模型");
  await api("/api/index", json("POST", {languages, force: false})); await refreshSystem();
}));
document.addEventListener("keydown", (event) => { if (event.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) && !$('dialog[open]')) { event.preventDefault(); $('#search-input').focus(); } });
renderHistory();
(async () => { try { await refreshInfo(); await loadPhotos(); await refreshSystem(); } catch (error) { notice(`连接未完成：${error.message}`); } })();
