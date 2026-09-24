import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Tauri expects a fixed dev port; see src-tauri/tauri.conf.json -> build.devUrl.
export default defineConfig({
  plugins: [react()],
  // Read VITE_* variables from the repo-root .env shared with the backend.
  envDir: '..',
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ['**/src-tauri/**'] },
  },
})
