import { defineConfig } from 'vite';

// Keep the model manifest and weights as inspectable, separately cached assets.
export default defineConfig({ build: { assetsInlineLimit: 0 } });
