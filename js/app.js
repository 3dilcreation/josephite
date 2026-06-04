/* ===================================================
   Vista360 – App Logic
   360° Photo: Pannellum  |  360° Video: A-Frame
   =================================================== */

'use strict';

/* ── Property Data ──────────────────────────────────────
   To add a real Insta360 photo tour:
     photo360: "images/your-equirectangular.jpg"
   To add a real Insta360 video tour:
     video360: "videos/your-360video.mp4"
   Both files must be equirectangular format.
   ────────────────────────────────────────────────────── */
const PROPERTIES = [
  {
    id: 1,
    title: "Sandton Summit Penthouse",
    type: "rental",
    badge: "For Rent",
    location: "Sandton, Johannesburg",
    price: "R 45 000",
    pricePeriod: "/month",
    beds: 4,
    baths: 3,
    sqft: "320 m²",
    parking: "2 covered bays",
    available: "Immediately",
    description: "An extraordinary sky-level residence atop one of Sandton's most prestigious towers. North-facing panoramic views stretching from the CBD to the Magaliesberg. Featuring custom Italian kitchen, heated floors throughout, a wraparound terrace, and a dedicated 4K home cinema room.",
    amenities: ["Swimming Pool", "24hr Security", "Gym", "Concierge", "Fibre Internet", "Backup Power", "Home Cinema", "Wine Cellar"],
    photo360: "images/property1-360.jpg",
    video360: "videos/property1-360.mp4",
    gradient: "linear-gradient(135deg, #0d1117 0%, #1a2a0d 40%, #2a3a1a 70%, #c9a227 100%)",
    accentColor: "#c9a227",
    icon: "fa-building"
  },
  {
    id: 2,
    title: "Clifton Cove Villa",
    type: "sale",
    badge: "For Sale",
    location: "Clifton, Cape Town",
    price: "R 22.5M",
    pricePeriod: "",
    beds: 5,
    baths: 4,
    sqft: "480 m²",
    parking: "3 garage bays",
    available: "April 2026",
    description: "A masterpiece of contemporary architecture perched above the legendary Clifton 4th Beach. Floor-to-ceiling glass invites the Atlantic Ocean into every room. Private infinity pool, bespoke teak decking, fully integrated smart home system, and a chef's kitchen by Bulthaup.",
    amenities: ["Ocean View", "Infinity Pool", "Smart Home", "Private Garden", "Wine Room", "Staff Quarters", "Outdoor Kitchen", "Direct Beach Access"],
    photo360: "images/property2-360.jpg",
    video360: "videos/property2-360.mp4",
    gradient: "linear-gradient(135deg, #0a1628 0%, #0d2a3a 40%, #1a3a50 70%, #00bcd4 100%)",
    accentColor: "#00bcd4",
    icon: "fa-water"
  },
  {
    id: 3,
    title: "Cape Quarter Heritage Loft",
    type: "rental",
    badge: "For Rent",
    location: "De Waterkant, Cape Town",
    price: "R 28 500",
    pricePeriod: "/month",
    beds: 2,
    baths: 2,
    sqft: "185 m²",
    parking: "1 basement bay",
    available: "Immediately",
    description: "A stunning double-volume loft conversion within a beautifully restored Victorian warehouse in the heart of De Waterkant. Exposed steel beams, polished concrete floors, and original brick walls frame a light-filled open-plan living space. Walking distance to Cape Quarter Square.",
    amenities: ["Double Volume", "Exposed Brick", "Roof Terrace", "Courtyard Access", "CCTV Security", "Fibre 200Mbps", "Bike Storage", "Pet Friendly"],
    photo360: "images/property3-360.jpg",
    video360: "videos/property3-360.mp4",
    gradient: "linear-gradient(135deg, #1a0a0a 0%, #2a1010 40%, #3a2010 70%, #e84040 100%)",
    accentColor: "#e84040",
    icon: "fa-landmark"
  },
  {
    id: 4,
    title: "Constantia Estate",
    type: "rental",
    badge: "For Rent",
    location: "Constantia, Cape Town",
    price: "R 62 000",
    pricePeriod: "/month",
    beds: 6,
    baths: 5,
    sqft: "780 m²",
    parking: "4 garage bays + visitors",
    available: "1 August 2026",
    description: "A sprawling family estate set on 2.4 hectares of lush gardens in the prestigious Constantia Valley, overlooked by the Constantiaberg mountains. This Georgian-style residence offers formal reception rooms, a farm-style kitchen, staff accommodation, a floodlit tennis court, and a resort-style pool.",
    amenities: ["Tennis Court", "Resort Pool", "2.4 Ha Garden", "Staff Quarters", "Wine Cellar", "Electric Fence", "Generator", "Borehole Water"],
    photo360: "images/property4-360.jpg",
    video360: "videos/property4-360.mp4",
    gradient: "linear-gradient(135deg, #0a1a0a 0%, #0d2010 40%, #1a3a1a 70%, #4caf50 100%)",
    accentColor: "#4caf50",
    icon: "fa-tree"
  }
];

let pannellumViewer = null;
let activePropertyId = null;
let activeTab = 'photo';

/* ── Render Property Cards ─────────────────────────────── */
function renderProperties(filter = 'all') {
  const grid = document.getElementById('propertiesGrid');
  if (!grid) return;

  const filtered = filter === 'all'
    ? PROPERTIES
    : PROPERTIES.filter(p => p.type === filter);

  grid.innerHTML = filtered.map(p => `
    <div class="property-card" data-id="${p.id}" onclick="openPhotoTour(${p.id})">
      <div class="card-media">
        <div class="card-thumb" style="background:${p.gradient};position:relative;overflow:hidden;">
          <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.5rem;">
            <i class="fas ${p.icon}" style="font-size:3rem;color:rgba(255,255,255,0.12);"></i>
            <div style="width:48px;height:48px;background:rgba(0,0,0,0.5);border:2px solid ${p.accentColor};border-radius:50%;display:flex;align-items:center;justify-content:center;">
              <i class="fas fa-eye" style="color:${p.accentColor};font-size:1.2rem;"></i>
            </div>
            <span style="font-size:0.72rem;color:rgba(255,255,255,0.5);letter-spacing:0.1em;text-transform:uppercase;">360° Panorama</span>
          </div>
        </div>
        <span class="card-badge ${p.type === 'sale' ? 'sale' : ''}">${p.badge}</span>
        <div class="card-360-icon"><i class="fas fa-street-view"></i></div>
        <div class="card-actions">
          <button class="card-action-btn btn-photo" onclick="event.stopPropagation();openPhotoTour(${p.id})">
            <i class="fas fa-camera"></i> 360° Photo
          </button>
          <button class="card-action-btn btn-video" onclick="event.stopPropagation();openVideoTour(${p.id})">
            <i class="fas fa-play"></i> 360° Video
          </button>
        </div>
      </div>
      <div class="card-body">
        <h3 class="card-title">${p.title}</h3>
        <p class="card-location"><i class="fas fa-map-marker-alt"></i>${p.location}</p>
        <div class="card-specs">
          <span class="card-spec"><i class="fas fa-bed"></i>${p.beds} Beds</span>
          <span class="card-spec"><i class="fas fa-bath"></i>${p.baths} Baths</span>
          <span class="card-spec"><i class="fas fa-ruler-combined"></i>${p.sqft}</span>
        </div>
        <div class="card-footer">
          <div class="card-price">${p.price}<span>${p.pricePeriod}</span></div>
          <button class="btn-outline-sm" onclick="event.stopPropagation();openPhotoTour(${p.id})">
            <i class="fas fa-vr-cardboard"></i> Tour
          </button>
        </div>
      </div>
    </div>
  `).join('');
}

/* ── Open Modal ────────────────────────────────────────── */
function openModal(type) {
  if (type === 'video-demo') {
    document.getElementById('videoDemoModal').classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

function openPhotoTour(id) {
  activePropertyId = id;
  const p = PROPERTIES.find(x => x.id === id);
  if (!p) return;
  populateModal(p);
  switchTab('photo');
  document.getElementById('tourModal').classList.add('active');
  document.body.style.overflow = 'hidden';
  setTimeout(() => initPannellum(p), 200);
}

function openVideoTour(id) {
  activePropertyId = id;
  const p = PROPERTIES.find(x => x.id === id);
  if (!p) return;
  populateModal(p);
  switchTab('video');
  document.getElementById('tourModal').classList.add('active');
  document.body.style.overflow = 'hidden';
  setTimeout(() => initVideoTour(p), 200);
}

function populateModal(p) {
  document.getElementById('modalTitle').textContent    = p.title;
  document.getElementById('modalLocation').innerHTML  = `<i class="fas fa-map-marker-alt"></i> ${p.location}`;
  document.getElementById('modalPrice').textContent   = p.price + p.pricePeriod;
  document.getElementById('detailBeds').textContent    = p.beds + ' Bedrooms';
  document.getElementById('detailBaths').textContent   = p.baths + ' Bathrooms';
  document.getElementById('detailSqft').textContent    = p.sqft;
  document.getElementById('detailParking').textContent = p.parking;
  document.getElementById('detailType').textContent    = p.badge;
  document.getElementById('detailAvail').textContent   = 'Available: ' + p.available;
  document.getElementById('detailDesc').textContent    = p.description;

  const amenitiesEl = document.getElementById('detailAmenities');
  amenitiesEl.innerHTML = p.amenities.map(a =>
    `<span class="amenity-tag">${a}</span>`
  ).join('');
}

/* ── Pannellum – 360° Photo ────────────────────────────── */
function initPannellum(p) {
  if (pannellumViewer) {
    try { pannellumViewer.destroy(); } catch(e) {}
    pannellumViewer = null;
  }

  const container = document.getElementById('panoramaViewer');
  container.innerHTML = '';

  /* Check if a real image file exists at the expected path.
     We use a fallback demo panorama so the viewer always initialises. */
  const imageUrl = p.photo360;

  /* Attempt to load the property's own image; on error fall back to demo */
  const img = new Image();
  img.onload = () => launchPannellum(imageUrl, p);
  img.onerror = () => launchPannellum(null, p);
  img.src = imageUrl;
}

function launchPannellum(imageUrl, p) {
  const container = document.getElementById('panoramaViewer');
  if (!container) return;

  /* Fallback equirectangular gradient canvas when no real image is available */
  if (!imageUrl) {
    container.innerHTML = `
      <div style="width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
                  background:${p.gradient};gap:1rem;">
        <i class="fas fa-camera-rotate" style="font-size:3rem;color:rgba(255,255,255,0.5);"></i>
        <p style="color:rgba(255,255,255,0.6);font-size:0.9rem;text-align:center;max-width:320px;line-height:1.6;">
          Place your Insta360 equirectangular image at<br>
          <code style="background:rgba(0,0,0,0.4);padding:0.2em 0.5em;border-radius:4px;color:#c9a227;">
            ${p.photo360}
          </code>
          <br>and it will load here automatically.
        </p>
      </div>`;
    return;
  }

  try {
    pannellumViewer = pannellum.viewer('panoramaViewer', {
      type: 'equirectangular',
      panorama: imageUrl,
      autoLoad: true,
      autoRotate: -1.5,
      compass: false,
      showZoomCtrl: false,
      showFullscreenCtrl: true,
      mouseZoom: true,
      friction: 0.4,
      hotSpots: [
        {
          pitch: 0, yaw: 90,
          type: 'info',
          text: p.title + ' – ' + p.location,
          cssClass: 'custom-hotspot'
        }
      ]
    });
  } catch(e) {
    container.innerHTML = `<div style="padding:2rem;color:#888;text-align:center;">360° viewer unavailable. Place equirectangular image at <code>${p.photo360}</code>.</div>`;
  }
}

/* ── A-Frame – 360° Video ──────────────────────────────── */
function initVideoTour(p) {
  const placeholder = document.getElementById('videoPlaceholder');
  placeholder.style.display = 'flex';

  /* If a real video path is provided, activate the A-Frame scene */
  if (p.video360 && p.video360 !== '') {
    const vid = document.getElementById('tour360video');
    if (vid) {
      vid.src = p.video360;
      vid.onerror = () => { /* keep placeholder visible */ };
      vid.oncanplay = () => {
        placeholder.style.display = 'none';
        const scene = document.getElementById('aframeScene');
        if (scene) {
          scene.style.cssText = 'height:480px;width:100%;visibility:visible;';
        }
        document.getElementById('videoControls').style.display = 'flex';
      };
    }
  }
}

function toggleVideo() {
  const vid = document.getElementById('tour360video');
  const btn = document.getElementById('playPauseBtn').querySelector('i');
  if (!vid) return;
  if (vid.paused) {
    vid.play();
    btn.className = 'fas fa-pause';
  } else {
    vid.pause();
    btn.className = 'fas fa-play';
  }
}

function toggleVideoFS() {
  const container = document.querySelector('.video360-container');
  if (!document.fullscreenElement) {
    container.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
}

/* Update video progress bar */
const vidEl = document.getElementById('tour360video');
if (vidEl) {
  vidEl.addEventListener('timeupdate', () => {
    if (vidEl.duration) {
      const pct = (vidEl.currentTime / vidEl.duration) * 100;
      const bar = document.getElementById('videoBar');
      if (bar) bar.style.width = pct + '%';
    }
  });
}

/* ── Tab Switching ─────────────────────────────────────── */
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  document.getElementById(tab + 'TabBtn').classList.add('active');
  document.getElementById(tab + 'Tab').classList.add('active');

  if (tab === 'photo' && activePropertyId) {
    const p = PROPERTIES.find(x => x.id === activePropertyId);
    if (p) setTimeout(() => initPannellum(p), 100);
  }
  if (tab === 'video' && activePropertyId) {
    const p = PROPERTIES.find(x => x.id === activePropertyId);
    if (p) initVideoTour(p);
  }
}

/* ── Close Modal ───────────────────────────────────────── */
function closeTourModal(event) {
  if (event.target === document.getElementById('tourModal')) closeTourModalDirect();
}

function closeTourModalDirect() {
  document.getElementById('tourModal').classList.remove('active');
  document.body.style.overflow = '';
  if (pannellumViewer) {
    try { pannellumViewer.destroy(); } catch(e) {}
    pannellumViewer = null;
  }
  const vid = document.getElementById('tour360video');
  if (vid) { vid.pause(); vid.src = ''; }
  const scene = document.getElementById('aframeScene');
  if (scene) scene.style.cssText = 'height:0;width:0;visibility:hidden;';
  document.getElementById('videoControls').style.display = 'none';
}

function closeVideoDemo(event) {
  if (event.target === document.getElementById('videoDemoModal')) {
    document.getElementById('videoDemoModal').classList.remove('active');
    document.body.style.overflow = '';
  }
}

/* ── Filter Buttons ────────────────────────────────────── */
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    renderProperties(this.dataset.filter);
  });
});

/* ── Navbar Scroll Effect ──────────────────────────────── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (window.scrollY > 60) nav.classList.add('scrolled');
  else nav.classList.remove('scrolled');
});

/* ── Hamburger Menu ────────────────────────────────────── */
document.getElementById('hamburger').addEventListener('click', function() {
  document.getElementById('navLinks').classList.toggle('open');
});

/* ── Animated Stats Counter ────────────────────────────── */
function animateCounters() {
  document.querySelectorAll('.big-num').forEach(el => {
    const target = parseInt(el.dataset.target, 10);
    const duration = 1800;
    const step = Math.ceil(target / (duration / 16));
    let current = 0;
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current.toLocaleString() + (target >= 100 ? '+' : '');
      if (current >= target) clearInterval(timer);
    }, 16);
  });
}

const statsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCounters();
      statsObserver.disconnect();
    }
  });
}, { threshold: 0.3 });

const statsBar = document.querySelector('.stats-bar');
if (statsBar) statsObserver.observe(statsBar);

/* ── Contact Form ──────────────────────────────────────── */
function handleFormSubmit(event) {
  event.preventDefault();
  const successEl = document.getElementById('formSuccess');
  successEl.classList.add('visible');
  event.target.reset();
  setTimeout(() => successEl.classList.remove('visible'), 5000);
}

/* ── Keyboard: Escape closes modal ────────────────────── */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeTourModalDirect();
    document.getElementById('videoDemoModal').classList.remove('active');
    document.body.style.overflow = '';
  }
});

/* ── Init ──────────────────────────────────────────────── */
renderProperties('all');
