// FILE: templates/client_actions.js (New File)
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Global State and Handles ---
    const appState = {
        workItems: {},      // The canonical { work_id: [assets] } map
        pivotedView: {},    // The derived { asset_type: [work_ids] } map
        currentGrouping: 'work_item',
        activeListItem: null, // Holds the currently selected DOM element in the list
    };

    const groupBySelect = document.getElementById('group-by-select');
    const filterInput = document.getElementById('filter-input');
    const primaryListContainer = document.getElementById('primary-list-container');
    const contentPane = document.getElementById('content-pane');

    // --- 2. Core Application Logic ---

    const main = async () => {
        setupEventListeners();
        try {
            const response = await fetch('/api/index');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            appState.workItems = data.work_items || {};
            renderPrimaryListView();
        } catch (error) {
            primaryListContainer.innerHTML = `<p class="placeholder">Failed to load dataset index. ${error.message}</p>`;
        }
    };

    const setupEventListeners = () => {
        groupBySelect.addEventListener('change', (e) => handleGroupingChange(e.target.value));
        filterInput.addEventListener('input', (e) => handleLocalFilter(e.target.value));
    };

    // --- 3. View Rendering and DOM Manipulation ---

    /**
     * (NEW UNIFIED HELPER) Takes a server-provided asset object and returns
     * the appropriate HTML string for its data representation.
     * @param {object} asset - The asset object with render_type and data.
     * @returns {string} The inner HTML for the asset.
     */
    const getAssetHTML = (asset) => {
        switch (asset.render_type) {
            case 'image':
                return `<img src="data:image/png;base64,${asset.data}" alt="asset image">`;
            case 'scalar':
                return `<div class="grid-item-scalar grid-item-data"><span class="scalar-value">${asset.data.value}</span></div>`;
            case 'vector':
                // This is the complete representation, now used everywhere.
                return `<div class="grid-item-vector grid-item-data">
                            <span class="vector-info">Shape: ${asset.data.shape}</span>
                            <span class="vector-info">Value: ${asset.data.repr}</span>
                        </div>`;
            case 'error':
                return `<div class="grid-item-error grid-item-data"><span class="error-info">${asset.data.message || 'Unknown Error'}</span></div>`;
            default:
                return `<div class="grid-item-data"><span class="error-info">?</span></div>`;
        }
    };


    const renderPrimaryListView = () => {
        primaryListContainer.innerHTML = '';
        contentPane.innerHTML = `<p class="placeholder">Select an item from the left to view details.</p>`;
        appState.activeListItem = null;

        if (appState.currentGrouping === 'work_item') {
            for (const workId in appState.workItems) {
                const count = appState.workItems[workId].length;
                const el = createListItem(workId, count, () => handleWorkItemSelection(workId, el));
                primaryListContainer.appendChild(el);
            }
        } else { // 'asset_type'
            pivotDataToAssetView();
            for (const assetType in appState.pivotedView) {
                const workIds = appState.pivotedView[assetType];
                const el = createListItem(assetType, workIds.length, () => handleAssetTypeSelection(assetType, workIds, el));
                primaryListContainer.appendChild(el);
            }
        }
    };

    const createGridItem = (workId, asset) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'grid-item';
        let contentHTML;

        // Special override for images in the grid view to enable lazy-loading.
        if (asset.render_type === 'image') {
            itemDiv.dataset.b64 = asset.data;
            contentHTML = `<img src="" alt="loading...">`; // Placeholder for IntersectionObserver
        } else {
            // For all other asset types, use the unified function directly.
            contentHTML = getAssetHTML(asset);
        }

        itemDiv.innerHTML = `${contentHTML}<div class="work-id-label">${workId.substring(0,12)}...</div>`;
        return itemDiv;
    };


    const createListItem = (label, count, onClick) => {
        const div = document.createElement('div');
        div.className = 'list-item';
        div.innerHTML = `${label} <span class="item-count">(${count})</span>`;
        div.onclick = onClick;
        return div;
    };

    const renderAssetDetailPane = (workId) => {
        const assets = appState.workItems[workId];
        contentPane.innerHTML = `<h3>Assets for ${workId.substring(0, 12)}...</h3><div class="asset-button-list"></div>`;
        const list = contentPane.querySelector('.asset-button-list');
        assets.forEach(assetType => {
            const button = document.createElement('button');
            button.textContent = assetType;
            button.onclick = () => fetchAndRenderSpecificAsset(workId, assetType);
            list.appendChild(button);
        });
    };
    
    const renderBatchAssetView = (assetType, batchData) => {
        contentPane.innerHTML = `<h3>All "${assetType}" Assets</h3><div class="asset-grid"></div>`;
        const grid = contentPane.querySelector('.asset-grid');

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    // Only act if it's an image that needs lazy-loading
                    if (el.dataset.b64) {
                        const img = el.querySelector('img');
                        img.src = `data:image/png;base64,${el.dataset.b64}`;
                        obs.unobserve(el);
                    }
                }
            });
        }, { rootMargin: "200px" });

        for (const workId in batchData.assets) {
            const asset = batchData.assets[workId];
            const itemDiv = createGridItem(workId, asset); // Use the new helper
            grid.appendChild(itemDiv);

            // Only observe the item if it's an image type
            if (asset.render_type === 'image') {
                observer.observe(itemDiv);
            }
        }
    };
    
    // --- 4. Event Handlers and Actions ---

    const handleGroupingChange = (newGrouping) => {
        appState.currentGrouping = newGrouping;
        renderPrimaryListView();
    };

    const handleLocalFilter = (filterText) => {
        const items = primaryListContainer.querySelectorAll('.list-item');
        const lowerFilterText = filterText.toLowerCase();
        items.forEach(item => {
            const itemText = item.textContent.toLowerCase();
            item.style.display = itemText.includes(lowerFilterText) ? '' : 'none';
        });
    };

    const updateActiveItem = (element) => {
        if (appState.activeListItem) {
            appState.activeListItem.classList.remove('selected');
        }
        element.classList.add('selected');
        appState.activeListItem = element;
    };
    
    const handleWorkItemSelection = (workId, element) => {
        updateActiveItem(element);
        renderAssetDetailPane(workId);
    };

    const handleAssetTypeSelection = (assetType, workIds, element) => {
        updateActiveItem(element);
        fetchAndRenderBatchAssets(assetType, workIds);
    };

    const pivotDataToAssetView = () => {
        appState.pivotedView = {};
        for (const workId in appState.workItems) {
            for (const assetType of appState.workItems[workId]) {
                if (!appState.pivotedView[assetType]) {
                    appState.pivotedView[assetType] = [];
                }
                appState.pivotedView[assetType].push(workId);
            }
        }
    };

    // --- 5. API Interaction ---

    const fetchAndRenderSpecificAsset = async (workId, assetType) => {
        contentPane.innerHTML = `<p class="loader">Loading ${assetType} for ${workId.substring(0,12)}...</p>`;
        try {
            const response = await fetch(`/api/asset/${workId}/${assetType}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const asset = await response.json();

            // Call the unified function to get the detail view's content.
            const detailContent = getAssetHTML(asset);

            contentPane.innerHTML = `
                <div class="asset-detail">
                    <h3>${assetType}</h3>
                    <h4>${workId}</h4>
                    ${detailContent}
                </div>`;
        } catch (error) {
            contentPane.innerHTML = `<p class="placeholder">Failed to load asset. ${error.message}</p>`;
        }
    };

    const fetchAndRenderBatchAssets = async (assetType, workIds) => {
        contentPane.innerHTML = `<p class="loader">Loading all ${workIds.length} "${assetType}" assets...</p>`;
        try {
            const response = await fetch('/api/batch/assets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asset_type: assetType, work_ids: workIds }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            renderBatchAssetView(assetType, data);
        } catch (error) {
             contentPane.innerHTML = `<p class="placeholder">Failed to load batch assets. ${error.message}</p>`;
        }
    };

    // --- Kick it off ---
    main();
});