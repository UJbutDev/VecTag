const LIGHT_STYLE = 'https://tiles.openfreemap.org/styles/liberty';
const DARK_STYLE = 'https://tiles.openfreemap.org/styles/dark';
const DEFAULT_PITCH = 62;

const ICONS = {
    sun: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    moon: '<svg class="icon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    mapPin: '<svg class="icon" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
    clock: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    eye: '<svg class="icon" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    ruler: '<svg class="icon" viewBox="0 0 24 24"><rect x="2" y="8" width="20" height="8" rx="1"/><line x1="6" y1="8" x2="6" y2="12"/><line x1="10" y1="8" x2="10" y2="12"/><line x1="14" y1="8" x2="14" y2="12"/><line x1="18" y1="8" x2="18" y2="12"/></svg>',
    gauge: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"/><line x1="12" y1="12" x2="16" y2="8"/><line x1="12" y1="12" x2="12" y2="9"/></svg>',
    alert: '<svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    check: '<svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    car: '<svg class="icon" viewBox="0 0 24 24" style="width:15px;height:15px;"><path d="M5 11l1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11"/><rect x="3" y="11" width="18" height="6" rx="2"/><circle cx="7.5" cy="17.5" r="1.5"/><circle cx="16.5" cy="17.5" r="1.5"/></svg>',
    twoWheeler: '<svg class="icon" viewBox="0 0 24 24" style="width:15px;height:15px;"><circle cx="5.5" cy="17.5" r="3.5"/><circle cx="18.5" cy="17.5" r="3.5"/><path d="M15 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM12 17.5V14l-3-3 4-3 2 3h3"/></svg>',
    backArrow: '<svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px;"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>'
};

function getVehicleIcon(vehicleType) {
    const t = (vehicleType || '').toLowerCase();
    if (t.includes('bike') || t.includes('scooter') || t.includes('two') || t.includes('motor')) {
        return ICONS.twoWheeler;
    }
    return ICONS.car;
}

const map = new maplibregl.Map({
    container: 'map',
    style: DARK_STYLE,
    center: [73.8567, 18.5204],
    zoom: 12,
    pitch: DEFAULT_PITCH,
    bearing: -15,
    antialias: true,
    cooperativeGestures: false
});

map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');

let markers = [];
let lastRouteGeometry = null;
let lastRouteIsApprox = false;

// Sidebar state kept module-level so the "Sightings" detail view can be
// shown/hidden without re-fetching or losing the current search's data.
let currentPlate = '';
let currentVehicleInfo = null;
let currentSortedPath = [];

function tiltUp() { map.setPitch(Math.min(map.getPitch() + 10, 75)); }
function tiltDown() { map.setPitch(Math.max(map.getPitch() - 10, 0)); }

function add3DBuildings() {
    if (map.getLayer('3d-buildings')) return;
    const isLight = document.body.classList.contains('light-mode');
    const colorStops = isLight
        ? [0, '#dfe3ea', 15, '#b9c2d6', 40, '#8b98b8', 100, '#5d6c94']
        : [0, '#2a3348', 15, '#374368', 40, '#4a5a8a', 100, '#6577b0'];

    let firstLabelLayerId;
    for (const layer of map.getStyle().layers) {
        if (layer.type === 'symbol') { firstLabelLayerId = layer.id; break; }
    }

    try {
        map.addLayer({
            id: '3d-buildings',
            source: 'openmaptiles',
            'source-layer': 'building',
            type: 'fill-extrusion',
            minzoom: 12.5,
            paint: {
                'fill-extrusion-color': ['interpolate', ['linear'], ['coalesce', ['get', 'render_height'], 8], 0, colorStops[1], 15, colorStops[3], 40, colorStops[5], 100, colorStops[7]],
                'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 8],
                'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
                'fill-extrusion-opacity': 0.9
            }
        }, firstLabelLayerId);
    } catch (e) { console.warn('buildings layer failed', e); }
}

function boostLabelContrast() {
    const isLight = document.body.classList.contains('light-mode');
    for (const layer of map.getStyle().layers) {
        if (layer.type !== 'symbol') continue;
        try {
            map.setPaintProperty(layer.id, 'text-color', isLight ? '#1a1a1a' : '#f5f5f5');
            map.setPaintProperty(layer.id, 'text-halo-color', isLight ? '#ffffff' : '#000000');
            map.setPaintProperty(layer.id, 'text-halo-width', 1.5);
        } catch (e) { /* icon-only layers skip safely */ }
    }
}

function applyLighting() {
    try {
        map.setLight({ anchor: 'viewport', color: '#ffffff', intensity: 0.55, position: [1.5, 90, 40] });
    } catch (e) { console.warn('lighting not supported', e); }
}

function dedupeConsecutive(coords) {
    return coords.filter((c, i) => {
        if (i === 0) return true;
        const prev = coords[i - 1];
        return c[0] !== prev[0] || c[1] !== prev[1];
    });
}

function distMeters(a, b) {
    const R = 6371000;
    const toRad = d => d * Math.PI / 180;
    const dLat = toRad(b[1] - a[1]);
    const dLon = toRad(b[0] - a[0]);
    const lat1 = toRad(a[1]);
    const lat2 = toRad(b[1]);
    const sinDLat = Math.sin(dLat / 2);
    const sinDLon = Math.sin(dLon / 2);
    const h = sinDLat * sinDLat + Math.cos(lat1) * Math.cos(lat2) * sinDLon * sinDLon;
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

function bearingDeg(a, b) {
    const toRad = d => d * Math.PI / 180;
    const toDeg = r => r * 180 / Math.PI;
    const lat1 = toRad(a[1]);
    const lat2 = toRad(b[1]);
    const dLon = toRad(b[0] - a[0]);
    const y = Math.sin(dLon) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
    const brng = toDeg(Math.atan2(y, x));
    return (brng + 360) % 360;
}

const MARKER_MERGE_THRESHOLD_M = 40;

// ---- Direction-indicator arrows (replaces the old dashed "flow" line) ----
// Draws small triangular icons at intervals along the actual road-snapped
// route, each rotated to match the real direction of travel at that point,
// and continuously animates them sliding along the line (looping) so the
// direction reads instantly even where the route crosses or backtracks
// over itself — a static dash pattern can't show that, but a rotated,
// moving arrow always points the correct way for that specific segment.
let arrowIntervalId = null;

function ensureArrowImage() {
    if (map.hasImage('route-arrow-icon')) return;

    // Rendered at 4x and scaled down via icon-size so edges stay crisp
    // (a small canvas drawn straight at native size looks jagged/cheap
    // once rotated by MapLibre).
    const scale = 3;
    const size = 40 * scale;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;

    // Soft drop shadow first, offset slightly, so the chevron reads as
    // sitting just above the route rather than flat-pasted onto it.
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = 10 * scale;
    ctx.shadowOffsetY = 3 * scale;

    // A single smooth chevron (open "V" pointing up), not a filled
    // triangle — reads as a direction indicator, not a game-asset arrow.
    ctx.beginPath();
    ctx.moveTo(cx - 11 * scale, 15 * scale);
    ctx.lineTo(cx, 4 * scale);
    ctx.lineTo(cx + 11 * scale, 15 * scale);
    ctx.lineTo(cx + 6 * scale, 15 * scale);
    ctx.lineTo(cx, 8.5 * scale);
    ctx.lineTo(cx - 6 * scale, 15 * scale);
    ctx.closePath();

    const grad = ctx.createLinearGradient(cx, 4 * scale, cx, 15 * scale);
    grad.addColorStop(0, '#fff7cc');
    grad.addColorStop(1, '#f6c945');
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();

    // Thin dark outline for definition against both light and dark tiles.
    ctx.beginPath();
    ctx.moveTo(cx - 11 * scale, 15 * scale);
    ctx.lineTo(cx, 4 * scale);
    ctx.lineTo(cx + 11 * scale, 15 * scale);
    ctx.lineTo(cx + 6 * scale, 15 * scale);
    ctx.lineTo(cx, 8.5 * scale);
    ctx.lineTo(cx - 6 * scale, 15 * scale);
    ctx.closePath();
    ctx.strokeStyle = 'rgba(20,15,0,0.55)';
    ctx.lineWidth = 1.2 * scale;
    ctx.stroke();

    const imgData = ctx.getImageData(0, 0, size, size);
    map.addImage('route-arrow-icon', { width: size, height: size, data: imgData.data }, { pixelRatio: scale });
}
function cumulativeDistances(coords) {
    const cum = [0];
    for (let i = 1; i < coords.length; i++) {
        cum.push(cum[i - 1] + distMeters(coords[i - 1], coords[i]));
    }
    return cum;
}

function pointAndBearingAtDistance(coords, cum, total, distIn) {
    const d = ((distIn % total) + total) % total;
    let i = 0;
    while (i < cum.length - 2 && cum[i + 1] < d) i++;
    const segStart = coords[i];
    const segEnd = coords[Math.min(i + 1, coords.length - 1)];
    const segLen = (cum[Math.min(i + 1, cum.length - 1)] - cum[i]) || 1;
    const t = Math.max(0, Math.min(1, (d - cum[i]) / segLen));
    const lon = segStart[0] + (segEnd[0] - segStart[0]) * t;
    const lat = segStart[1] + (segEnd[1] - segStart[1]) * t;
    const bearing = bearingDeg(segStart, segEnd);
    return { lon, lat, bearing };
}

function buildArrowOffsets(total) {
    const desired = Math.max(5, Math.min(16, Math.round(total / 400)));
    const spacing = total / desired;
    const offsets = [];
    for (let i = 0; i < desired; i++) offsets.push(i * spacing);
    return offsets;
}

function buildArrowFeatureCollection(coords, cum, total, offsets, animOffset) {
    const features = offsets.map(base => {
        const { lon, lat, bearing } = pointAndBearingAtDistance(coords, cum, total, base + animOffset);
        return { type: 'Feature', properties: { bearing }, geometry: { type: 'Point', coordinates: [lon, lat] } };
    });
    return { type: 'FeatureCollection', features };
}

function stopArrowAnimation() {
    if (arrowIntervalId) {
        clearInterval(arrowIntervalId);
        arrowIntervalId = null;
    }
}

function clearArrowLayer() {
    stopArrowAnimation();
    try {
        if (map.getLayer('route-arrows-anim')) map.removeLayer('route-arrows-anim');
        if (map.getSource('route-arrows-anim-src')) map.removeSource('route-arrows-anim-src');
    } catch (e) { /* style may be mid-swap */ }
}

function startArrowAnimation(coords) {
    clearArrowLayer();
    if (!coords || coords.length < 2) return;

    ensureArrowImage();

    const cum = cumulativeDistances(coords);
    const total = cum[cum.length - 1];
    if (!total || total < 30) return; // too short to animate meaningfully

    const offsets = buildArrowOffsets(total);

    try {
        map.addSource('route-arrows-anim-src', {
            type: 'geojson',
            data: buildArrowFeatureCollection(coords, cum, total, offsets, 0)
        });
        map.addLayer({
            id: 'route-arrows-anim',
            type: 'symbol',
            source: 'route-arrows-anim-src',
            layout: {
                'icon-image': 'route-arrow-icon',
                'icon-size': 0.55,
                'icon-rotate': ['get', 'bearing'],
                'icon-rotation-alignment': 'map',
                'icon-allow-overlap': true,
                'icon-ignore-placement': true
            },
            paint: {
                'icon-opacity': 0.92
            }
        });
    } catch (e) {
        console.warn('Failed to start arrow animation:', e);
        return;
    }

    const TRAVERSE_DURATION_MS = 120000; // fixed feel regardless of route length
    const STEP_MS = 100;
    const stepDistance = total / (TRAVERSE_DURATION_MS / STEP_MS);
    let animOffset = 0;

    arrowIntervalId = setInterval(() => {
        const src = map.getSource('route-arrows-anim-src');
        if (!src) { stopArrowAnimation(); return; }
        animOffset = (animOffset + stepDistance) % total;
        try {
            src.setData(buildArrowFeatureCollection(coords, cum, total, offsets, animOffset));
        } catch (e) { /* source may be mid-removal during a redraw */ }
    }, STEP_MS);
}
// ---- end direction-indicator arrows ----

function clearRoute() {
    clearArrowLayer();
    try {
        ['route-glow-outer', 'route-glow-inner', 'route-core', 'route-casing'].forEach(id => {
            if (map.getLayer(id)) map.removeLayer(id);
        });
        if (map.getSource('route')) map.removeSource('route');
    } catch (e) {
        console.warn('clearRoute failed (style may not be ready yet):', e);
    }
    lastRouteGeometry = null;
    lastRouteIsApprox = false;
}

function drawRoute(geometry, opts = {}) {
    const approx = !!opts.approx;

    if (!geometry || !geometry.coordinates || geometry.coordinates.length < 2) {
        console.warn('drawRoute received no usable geometry — leaving existing route untouched');
        return;
    }

    if (!map.isStyleLoaded()) {
        console.warn('Map style not ready yet — deferring route draw until idle');
        map.once('idle', () => drawRoute(geometry, opts));
        return;
    }

    clearArrowLayer();
    ['route-glow-outer', 'route-glow-inner', 'route-core', 'route-casing'].forEach(id => {
        if (map.getLayer(id)) map.removeLayer(id);
    });
    if (map.getSource('route')) map.removeSource('route');

    const isLight = document.body.classList.contains('light-mode');
    const dash = approx ? [1, 1.6] : undefined;

    try {
        map.addSource('route', { type: 'geojson', data: { type: 'Feature', geometry, properties: {} } });

        if (isLight) {
            map.addLayer({
                id: 'route-casing', type: 'line', source: 'route',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': '#ffffff', 'line-width': 11, 'line-opacity': approx ? 0.7 : 1,
                    ...(dash ? { 'line-dasharray': dash } : {})
                }
            });
            map.addLayer({
                id: 'route-core', type: 'line', source: 'route',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': '#db2777', 'line-width': 6, 'line-opacity': approx ? 0.85 : 1,
                    ...(dash ? { 'line-dasharray': dash } : {})
                }
            });
        } else {
            map.addLayer({
                id: 'route-glow-outer', type: 'line', source: 'route',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: { 'line-color': '#22d3ee', 'line-width': 16, 'line-blur': 8, 'line-opacity': approx ? 0.2 : 0.35 }
            });
            map.addLayer({
                id: 'route-glow-inner', type: 'line', source: 'route',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: { 'line-color': '#38bdf8', 'line-width': 9, 'line-blur': 3, 'line-opacity': approx ? 0.35 : 0.55 }
            });
            map.addLayer({
                id: 'route-core', type: 'line', source: 'route',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': '#e0f7ff', 'line-width': 3, 'line-opacity': approx ? 0.8 : 1,
                    ...(dash ? { 'line-dasharray': dash } : {})
                }
            });
        }

        if (!approx) {
            startArrowAnimation(geometry.coordinates);
        }

        console.log(`Route drawn (${approx ? 'approx straight-line' : 'real route'}) with`, geometry.coordinates.length, 'points');
    } catch (e) {
        console.error('Failed to draw route layers, retrying once map is idle:', e);
        map.once('idle', () => drawRoute(geometry, opts));
    }
}

function redrawAll() {
    add3DBuildings();
    boostLabelContrast();
    drawRoute(lastRouteGeometry, { approx: lastRouteIsApprox });
    applyLighting();
}

map.on('load', redrawAll);

function toggleTheme() {
    const isLight = document.body.classList.toggle('light-mode');
    document.getElementById('themeToggle').innerHTML = isLight ? ICONS.moon : ICONS.sun;
    map.setStyle(isLight ? LIGHT_STYLE : DARK_STYLE);
    map.once('styledata', redrawAll);
}

async function fetchRoute(coordsLonLat) {
    try {
        const coordStr = coordsLonLat.map(c => c.join(',')).join(';');
        const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('OSRM request failed: ' + res.status);
        const data = await res.json();
        if (data.routes && data.routes.length > 0) {
            return data.routes[0].geometry;
        }
        throw new Error('No route returned, code was: ' + data.code);
    } catch (e) {
        console.warn('Routing failed, falling back to straight line. Reason:', e.message);
        return { type: 'LineString', coordinates: coordsLonLat };
    }
}

function formatDateTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
}

function renderSidebar(plate, vehicleInfo, sortedPath) {
    currentPlate = plate;
    currentVehicleInfo = vehicleInfo;
    currentSortedPath = sortedPath || [];

    const container = document.getElementById('sidebarContent');

    if (!vehicleInfo) {
        container.innerHTML = `
      <div class="empty-state">
        <svg class="icon" viewBox="0 0 24 24" style="display:block;margin:0 auto 10px;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        No detections found for "${plate}".
      </div>`;
        return;
    }

    const status = (vehicleInfo.status || 'unknown').toLowerCase();
    const statusClass = status === 'stolen' ? 'status-stolen' : status === 'blacklisted' ? 'status-blacklisted' : 'status-normal';
    const statusIcon = status === 'normal' ? ICONS.check : ICONS.alert;
    const vehicleIcon = getVehicleIcon(vehicleInfo.vehicle_type);
    const pathLength = currentSortedPath.length;

    container.innerHTML = `
    <div class="plate-title">${plate}</div>
    <div class="owner-line">${ICONS.mapPin.replace('class="icon"', 'class="icon" style="width:13px;height:13px;"')} ${vehicleInfo.owner_name} · ${vehicleInfo.vehicle_type} ${vehicleIcon}</div>
    <div class="status-badge ${statusClass}">${statusIcon} ${status}</div>

    <div class="kpi-grid">
      <div class="kpi-card clickable" onclick="showSightingsDetail()">${ICONS.eye}<div class="kpi-value">${pathLength}</div><div class="kpi-label">Sightings</div></div>
      <div class="kpi-card">${ICONS.ruler}<div class="kpi-value">${vehicleInfo.total_distance_km}</div><div class="kpi-label">Km traveled</div></div>
      <div class="kpi-card">${ICONS.gauge}<div class="kpi-value">${vehicleInfo.avg_speed_kmph !== null ? vehicleInfo.avg_speed_kmph : '—'}</div><div class="kpi-label">Avg km/h</div></div>
    </div>

    <div class="info-row"><span class="info-label">${ICONS.mapPin} Start location</span><span class="info-value">${vehicleInfo.start_location || '—'}</span></div>
    <div class="info-row"><span class="info-label">${ICONS.mapPin} End location</span><span class="info-value">${vehicleInfo.end_location || '—'}</span></div>
    <div class="info-row"><span class="info-label">${ICONS.clock} First seen</span><span class="info-value">${formatDateTime(vehicleInfo.first_seen)}</span></div>
    <div class="info-row"><span class="info-label">${ICONS.clock} Last seen</span><span class="info-value">${formatDateTime(vehicleInfo.last_seen)}</span></div>
  `;
}

function showSightingsDetail() {
    const container = document.getElementById('sidebarContent');
    const lastIndex = currentSortedPath.length - 1;

    const rows = currentSortedPath.map((p, i) => {
        let badge = '';
        if (i === 0) badge = '<span class="sighting-badge" style="background:rgba(74,222,128,0.15);color:#4ade80;">Start</span>';
        else if (i === lastIndex) badge = '<span class="sighting-badge" style="background:rgba(248,113,113,0.15);color:#f87171;">Last</span>';

        return `
      <div class="sighting-row">
        <div class="cam-name">${p.camera_name} ${badge}</div>
        <div class="sighting-time">${formatDateTime(p.time)}</div>
      </div>`;
    }).join('');

    container.innerHTML = `
    <div class="sightings-detail-header">
      <button class="back-btn" onclick="renderSidebar(currentPlate, currentVehicleInfo, currentSortedPath)">${ICONS.backArrow} Back</button>
      <div class="sightings-detail-title">All sightings (${currentSortedPath.length})</div>
    </div>
    ${rows}
  `;
}

async function loadTrajectory() {
    const plate = document.getElementById('plateInput').value.trim();
    if (!plate) { alert('Enter a plate number'); return; }

    const res = await fetch(`http://localhost:8000/trajectory/${plate}`);
    const data = await res.json();

    markers.forEach(m => m.remove());
    markers = [];
    clearRoute();

    const sortedPath = data.path ? [...data.path].sort((a, b) => new Date(a.time) - new Date(b.time)) : [];
    renderSidebar(plate, data.vehicle_info, sortedPath);

    if (sortedPath.length === 0) return;

    const lastIndex = sortedPath.length - 1;
    const coordsLonLat = sortedPath.map(p => [p.longitude, p.latitude]);

    sortedPath.forEach((p, i) => {
        const isStart = i === 0;
        const isEnd = i === lastIndex;
        const coord = [p.longitude, p.latitude];
        const startCoord = coordsLonLat[0];
        const endCoord = coordsLonLat[lastIndex];
        const startEndCoincide = lastIndex > 0 && distMeters(startCoord, endCoord) < MARKER_MERGE_THRESHOLD_M;
        const bothEndsHere = startEndCoincide && (isStart || isEnd);

        if (!isStart && !isEnd) {
            const nearStart = distMeters(coord, coordsLonLat[0]) < MARKER_MERGE_THRESHOLD_M;
            const nearEnd = distMeters(coord, coordsLonLat[lastIndex]) < MARKER_MERGE_THRESHOLD_M;
            if (nearStart || nearEnd) return;
        }
        if (bothEndsHere && isEnd) return;

        const color = bothEndsHere ? '#f59e0b' : (isStart ? '#2ecc71' : (isEnd ? '#ef4444' : '#38bdf8'));
        const label = bothEndsHere ? 'Start & Last seen (same location)' : (isStart ? 'Start' : (isEnd ? 'Last seen' : `Stop #${i + 1}`));
        const el = document.createElement('div');

        if (isStart || isEnd) {
            el.className = 'pulse-marker';
            el.style.zIndex = isStart ? '3' : '2';
            const ring = document.createElement('div');
            ring.className = 'pulse-ring';
            ring.style.background = color;
            const dot = document.createElement('div');
            dot.className = 'pulse-dot';
            dot.style.background = color;
            el.appendChild(ring);
            el.appendChild(dot);
        } else {
            el.style.width = '13px';
            el.style.height = '13px';
            el.style.borderRadius = '50%';
            el.style.background = color;
            el.style.border = '2px solid #fff';
            el.style.boxShadow = `0 0 0 4px ${color}55, 0 2px 8px rgba(0,0,0,0.6)`;
            el.style.zIndex = '1';
        }

        const popup = new maplibregl.Popup({ offset: 16 }).setHTML(
            `<b>${p.camera_name}</b><br>${new Date(p.time).toLocaleString()}<br>${label}`
        );

        markers.push(new maplibregl.Marker({ element: el }).setLngLat([p.longitude, p.latitude]).setPopup(popup).addTo(map));
    });

    const dedupedCoords = dedupeConsecutive(coordsLonLat);

    if (dedupedCoords.length > 1) {
        const routeGeometry = await fetchRoute(dedupedCoords);
        lastRouteGeometry = routeGeometry;
        lastRouteIsApprox = false;
        drawRoute(routeGeometry, { approx: false });
    } else {
        lastRouteGeometry = null;
        lastRouteIsApprox = false;
        drawRoute(null);
    }

    const bounds = coordsLonLat.reduce((b, c) => b.extend(c), new maplibregl.LngLatBounds(coordsLonLat[0], coordsLonLat[0]));

    const spanKm = (data.vehicle_info && data.vehicle_info.total_distance_km) || 0;
    const fitPitch = spanKm > 15 ? 30 : DEFAULT_PITCH;

    map.fitBounds(bounds, {
        padding: { top: 100, bottom: 100, left: 100, right: 400 },
        pitch: fitPitch,
        duration: 1400,
        essential: true
    });
}