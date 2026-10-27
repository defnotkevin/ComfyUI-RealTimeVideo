import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

app.registerExtension({
    name: 'Krea.RealtimeStudio',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'KreaRTLiveStudio') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.addWidget('button', 'Open Live Studio', null, () => {
                const url = new URL(api.apiURL('/krea-live/studio'), location.origin);
                // Fragment settings stay in the browser; no prompt in server access logs.
                if (this.kreaSettings) url.hash = encodeURIComponent(JSON.stringify(this.kreaSettings));
                window.open(url.href, '_blank', 'noopener');
            });
            this.setSize([350, 120]);
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            this.kreaSettings = message?.krea_settings?.[0];
        };
    }
});
