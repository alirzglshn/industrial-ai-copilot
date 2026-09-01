/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#818cf8",
          hover: "#a5b4fc",
        },
      },
    },
  },
  plugins: [],
};
