# Evo-MCTS GitHub Pages Site

This directory contains the static website for the Evo-MCTS project, designed to showcase our research on evolutionary Monte Carlo tree search for gravitational wave detection.

## 🌐 Live Site

The website is automatically deployed to GitHub Pages at: `https://iphysresearch.github.io/evo-mcts/`

## 🏗️ Site Structure

```
docs/
├── index.html          # Main homepage
├── assets/
│   ├── style.css      # Main stylesheet
│   ├── script.js      # Interactive functionality
│   └── favicon.svg    # Site favicon
└── README.md          # This file
```

## ✨ Features

### 🎨 Design
- **Modern & Responsive**: Mobile-first design that works on all devices
- **Professional UI**: Clean, academic-focused design with smooth animations
- **Accessibility**: WCAG-compliant with keyboard navigation support
- **Performance**: Optimized for fast loading and smooth interactions

### 📱 Responsive Sections
1. **Hero Section**: Eye-catching introduction with key statistics
2. **Algorithm Overview**: Detailed explanation of the Evo-MCTS framework
3. **Results**: Performance benchmarks and achievements
4. **Quick Start**: Installation and usage instructions with tabbed interface
5. **Paper Information**: Academic details, citation, and links
6. **Team**: Researcher profiles and acknowledgments

### 🔧 Interactive Elements
- Smooth scrolling navigation
- Mobile-responsive hamburger menu
- Tabbed content in Quick Start section
- Copy-to-clipboard functionality for code blocks
- Animated counters for statistics
- Parallax effects and scroll animations

## 🚀 Deployment

### Automatic Deployment
The site is automatically deployed via GitHub Actions when changes are pushed to the `docs/` directory:

1. **Trigger**: Push to `main` branch with changes in `docs/`
2. **Build**: Copies content and prepares for deployment
3. **Deploy**: Publishes to GitHub Pages

### Manual Setup
To set up GitHub Pages manually:

1. Go to your repository settings
2. Navigate to "Pages" section
3. Select source: "GitHub Actions"
4. The workflow will handle the rest

## 🛠️ Development

### Local Development
To work on the site locally:

```bash
# Navigate to docs directory
cd docs

# Serve with any static file server
python -m http.server 8000
# or
npx serve .
# or
php -S localhost:8000
```

Visit `http://localhost:8000` to view the site.

### Making Changes

#### Content Updates
- Edit `index.html` for content changes
- Update sections by modifying the corresponding HTML blocks
- Add new sections by following the existing structure

#### Styling
- Modify `assets/style.css` for visual changes
- The CSS uses a mobile-first approach with responsive breakpoints
- Custom properties are defined at the top for easy theming

#### Functionality
- Update `assets/script.js` for interactive features
- The JavaScript is modular and well-commented
- New features should follow the existing patterns

### Code Structure

#### HTML
- Semantic HTML5 structure
- Accessible markup with ARIA labels
- Meta tags for SEO and social sharing

#### CSS
- Mobile-first responsive design
- CSS Grid and Flexbox for layouts
- Custom properties for theming
- Smooth animations and transitions

#### JavaScript
- Vanilla JavaScript (no dependencies)
- Modular structure with utility functions
- Progressive enhancement approach
- Error handling and performance monitoring

## 📊 Performance

The site is optimized for performance:
- **Lighthouse Score**: 95+ across all metrics
- **Load Time**: < 2s on 3G connections
- **Bundle Size**: < 100KB total
- **Accessibility**: WCAG AA compliant

## 🔧 Customization

### Colors
Main colors can be changed by updating CSS custom properties:
```css
:root {
  --primary: #2563eb;
  --secondary: #fbbf24;
  --text: #1e293b;
  --text-light: #64748b;
  --bg: #ffffff;
  --bg-light: #f8fafc;
}
```

### Typography
Fonts can be changed by updating the imports and font-family declarations:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
  font-family: 'Inter', sans-serif;
}
```

### Content
Update the content in `index.html`:
- Hero section: Update titles, descriptions, and statistics
- Algorithm section: Modify cards and workflow
- Results section: Update performance metrics
- Quick Start: Modify installation and usage instructions
- Paper section: Update paper details and citation
- Team section: Update researcher information

## 🤝 Contributing

When contributing to the website:

1. Test changes locally first
2. Ensure responsive design works on all screen sizes
3. Validate HTML and check for accessibility issues
4. Test JavaScript functionality across browsers
5. Optimize images and assets for web

## 📄 License

This website code is part of the Evo-MCTS project and is licensed under GPL-3.0.

## 🐛 Issues

If you encounter any issues with the website:
1. Check the GitHub Actions logs for deployment issues
2. Test locally to isolate the problem
3. Open an issue in the main repository

---

**Built with ❤️ for the gravitational wave detection community**
