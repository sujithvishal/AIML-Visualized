# AI Learning Hub

An interactive AI & Machine Learning learning platform built with React and Vite. Learn ML and Deep Neural Networks through animated diagrams, interactive playgrounds, Python code examples, and knowledge-check quizzes — not static text.

![AI Learning Hub](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-38BDF8?logo=tailwindcss&logoColor=white) ![Framer Motion](https://img.shields.io/badge/Framer_Motion-13-FF0055?logo=framer&logoColor=white)

---

## What's inside

### Sections

| Section | Status | Topics |
|---------|--------|--------|
| **Machine Learning** | ✅ Active | What is ML?, Linear Regression, Logistic Regression |
| **Deep Neural Networks** | ✅ Active | What is DNN?, Perceptron, Multi-Layer Perceptron |
| **NLP** | 🔒 Coming soon | — |
| **LLM** | 🔒 Coming soon | — |

### Features

- **Interactive playgrounds** — drag scatter points, move decision boundaries, tune sliders and watch graphs update live
- **Animated diagrams** — data flow through ML pipelines, neural network forward propagation, sigmoid curves, cost surfaces
- **Canvas visualisations** — 2D classifier boundaries, activation function graphs, perceptron diagrams, MLP layer animations
- **Python code examples** — syntax-highlighted, copyable scikit-learn and TensorFlow snippets for every topic
- **Backpropagation visualizer** — step-through chain rule breakdown with per-step explanations
- **Knowledge-check quizzes** — 5 MCQs per topic with instant feedback, explanations, and XP awards
- **Gamification** — XP points, level bar, and badges (ML Beginner, Regression Explorer, Neural Network Starter)
- **Deep Dive sections** — collapsible optional content for learners who want to go further
- **Global search** — ⌘K search across all topics and concepts
- **Dark / Light theme** — persisted to localStorage
- **Responsive layout** — collapsible sidebar becomes a drawer on mobile

---

## Tech stack

| Tool | Version | Purpose |
|------|---------|---------|
| [React](https://react.dev) | 19 | UI framework |
| [Vite](https://vite.dev) | 8 | Build tool & dev server |
| [Tailwind CSS](https://tailwindcss.com) | 4 | Utility-first styling (`@tailwindcss/vite` plugin) |
| [Framer Motion](https://motion.dev) | 13 | Animations & page transitions |
| [Lucide React](https://lucide.dev) | latest | Icons |
| Canvas 2D API | native | Interactive graphs and neural network diagrams |

---

## Getting started

### Prerequisites

- **Node.js** ≥ 18
- **npm** ≥ 9 (comes with Node)

### Install and run

```bash
# 1. Clone the repo
git clone https://github.com/vsujithvishal/aiml-visualized.git
cd aiml-visualized

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

The app will be available at **http://localhost:5173**

---

## Available commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start the local development server with HMR |
| `npm run build` | Build optimised production bundle to `dist/` |
| `npm run preview` | Serve the production build locally for testing |
| `npm run lint` | Run ESLint across the entire codebase |

---

## Project structure

```
src/
├── context/
│   ├── topics.js              # ML + DNN topic definitions
│   ├── AppContextInstance.js  # React context object
│   ├── useApp.js              # useApp() hook
│   └── AppContext.jsx         # AppProvider (theme, XP, progress, badges)
│
├── components/
│   ├── Navbar.jsx             # Sticky glassmorphism navbar
│   ├── Sidebar.jsx            # Collapsible sidebar with search & progress
│   ├── CodeBlock.jsx          # Python syntax highlighter with copy button
│   ├── Quiz.jsx               # MCQ quiz engine with XP awards
│   ├── XPPanel.jsx            # Level bar + badge showcase
│   ├── GlobalSearch.jsx       # ⌘K search overlay
│   ├── ParticleBackground.jsx # Animated canvas particle background
│   ├── TheoryBlock.jsx        # Card-grid theory section
│   ├── Callout.jsx            # Callout boxes (info, formula, warning, etc.)
│   └── DeepDive.jsx           # Collapsible optional deep-dive sections
│
└── pages/
    ├── ml/
    │   ├── WhatIsML.jsx           # What is Machine Learning?
    │   ├── LinearRegression.jsx   # Linear Regression
    │   └── LogisticRegression.jsx # Logistic Regression
    └── dnn/
        ├── WhatIsDNN.jsx          # What is a Deep Neural Network?
        ├── Perceptron.jsx         # Perceptron
        └── MLP.jsx                # Multi-Layer Perceptron
```

---

## Author

**Sujith Vishal** — [linkedin.com/in/sujith-vishal](https://www.linkedin.com/in/sujith-vishal/)
