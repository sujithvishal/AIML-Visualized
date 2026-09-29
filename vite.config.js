import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/AIML-Visualized/',
  plugins: [react(), tailwindcss()],
})
