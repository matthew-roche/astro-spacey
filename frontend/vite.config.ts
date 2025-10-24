import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  console.log('Backend URL:', env.VITE_API_URL);

  return {
    plugins: [react(), tailwindcss()],
    root: '.',
    server: {
      proxy: {
        '^/api(/.*)?': {
          target: env.VITE_API_URL,
          changeOrigin: true,
          //rewrite: p => p.replace(/^\/api/, ''),
          headers: { 'X-Demo-Secret': env.VITE_API_SEC }
        }
      }
    }
  };
});