/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        barlow: ['"Barlow"', "sans-serif", "Philosopher"],
      },
    },
    screens: {
      smCustom: "710px",
      sm: "640px",
      md: "768px",
      lg: "1024px",
      xl: "1280px",
      // => @media (min-width: 640px) { ... }
    },
  },
  plugins: [],
};
