/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'sentinel-blue': '#0066CC',
        'sentinel-green': '#00AA55',
        'sentinel-red': '#DD2C00',
        'sentinel-yellow': '#FF9800',
      },
    },
  },
  plugins: [],
}

