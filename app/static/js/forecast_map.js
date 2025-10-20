document.addEventListener('DOMContentLoaded', () => {
    const latInput = document.getElementById('latitude');
    const lonInput = document.getElementById('longitude');
    const boroughSelect = document.getElementById('borough');
    
    // Coordenadas centrales aproximadas de los distritos
    const boroughCoords = {
        'NYC': [40.7128, -74.0060],
        'Manhattan': [40.7831, -73.9712],
        'Brooklyn': [40.6782, -73.9442],
        'Queens': [40.7282, -73.7949],
        'Bronx': [40.8448, -73.8648],
        'Staten Island': [40.5795, -74.1502]
    };

    // Inicializar mapa usando las coordenadas iniciales del formulario
    let currentLat = parseFloat(latInput.value) || boroughCoords['NYC'][0];
    let currentLon = parseFloat(lonInput.value) || boroughCoords['NYC'][1];
    const map = L.map('mini-map').setView([currentLat, currentLon], 12);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { // Tile más limpio
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Marcador arrastrable
    let marker = L.marker([currentLat, currentLon], { draggable: true }).addTo(map);

    // Función para actualizar los inputs
    function updateInputs(latlng) {
        latInput.value = latlng.lat.toFixed(5);
        lonInput.value = latlng.lng.toFixed(5);
    }

    // Eventos del mapa
    map.on('click', (e) => {
        marker.setLatLng(e.latlng);
        updateInputs(e.latlng);
        boroughSelect.value = 'NYC'; // Resetea el selector de distrito si hacen clic
    });

    marker.on('dragend', () => {
        const latlng = marker.getLatLng();
        updateInputs(latlng);
        boroughSelect.value = 'NYC'; // Resetea el selector de distrito si mueven el marcador
    });

    // Evento del selector de distrito
    boroughSelect.addEventListener('change', () => {
        const selectedBorough = boroughSelect.value;
        if (boroughCoords[selectedBorough]) {
            const coords = boroughCoords[selectedBorough];
            map.setView(coords, 12);
            marker.setLatLng(coords);
            updateInputs(L.latLng(coords[0], coords[1]));
        }
    });

    // Inicializar inputs con los valores del marcador
    updateInputs(marker.getLatLng());
});