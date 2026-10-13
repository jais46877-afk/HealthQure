# 🎨 UI/UX Design System Specification: "ui-ux-pro-max" Reference Architecture

**Project**: Hybrid Quantum-Classical Multi-Disease Diagnostic Platform  
**Design Reference**: Healthcare Enterprise Portal (Mediva Pharmaceuticals & CarePlus Medical Architecture)  
**Version**: 2.0  
**Themes Supported**: Full Dual-Engine Theme (Light Mode Default + Obsidian Cyan Dark Mode)

---

## 1. Executive Summary & Design Philosophy
This design system adapts modern, patient-first healthcare web architecture to high-performance quantum clinical diagnostic software. It eliminates flat, harsh interfaces in favor of **soft depth, rounded squircle containers, clear visual hierarchy, and mathematically strict 8px spatial rhythm**.

---

## 2. 4-Corner Professional Constraints

### Corner 1: Color Palette & Glassmorphic Depth

#### Light Mode Palette (Default — Modeled on Mediva / CarePlus Reference)
- **Base Background (`--bg-primary`)**: `#FFFFFF` (Pure clinical clean white)
- **Section & Canvas Background (`--bg-secondary`)**: `#F4F8FA` to `#EBF3F6` (Ultra-soft ice-blue / teal-tinted wash)
- **Elevated Card Surface (`--bg-card`)**: `#FFFFFF` (Solid white with micro-border)
- **Glassmorphic Surface (`--bg-glass`)**: `rgba(255, 255, 255, 0.85)` with `backdrop-filter: blur(16px)`
- **Primary Healthcare Teal Accent (`--accent`)**: `#10758F` / `#1A8099` (Crisp, authoritative deep cyan-teal)
- **Primary Accent Hover (`--accent-hover`)**: `#0D5D73`
- **Soft Accent Wash (`--accent-soft`)**: `rgba(16, 117, 143, 0.08)`
- **Accent Glow (`--accent-glow`)**: `rgba(16, 117, 143, 0.18)`
- **Teal Gradient Banner**: `linear-gradient(135deg, #10758F 0%, #1A8099 50%, #2094B0 100%)`
- **Borders (`--border`)**: `rgba(16, 117, 143, 0.12)`
- **Subtle Borders (`--border-subtle`)**: `rgba(0, 0, 0, 0.06)`
- **Text Primary (`--text-primary`)**: `#111827` (Deep clinical slate)
- **Text Secondary (`--text-secondary`)**: `#4B5563` (Neutral gray)
- **Text Muted (`--text-muted`)**: `#9CA3AF`

#### Dark Mode Palette (Obsidian & Electric Cyan)
- **Base Background (`--bg-primary`)**: `#0B111E` (Deep obsidian slate, never harsh #000)
- **Section & Canvas Background (`--bg-secondary`)**: `#0F172A` (Midnight canvas)
- **Elevated Card Surface (`--bg-card`)**: `#151E2E` (Deep surface slate)
- **Glassmorphic Surface (`--bg-glass`)**: `rgba(21, 30, 46, 0.80)` with `backdrop-filter: blur(16px)`
- **Primary Accent (`--accent`)**: `#22D3EE` (Vivid electric cyan)
- **Primary Accent Hover (`--accent-hover`)**: `#06B6D4`
- **Soft Accent Wash (`--accent-soft`)**: `rgba(34, 211, 238, 0.12)`
- **Accent Glow (`--accent-glow`)**: `rgba(34, 211, 238, 0.25)`
- **Borders (`--border`)**: `rgba(34, 211, 238, 0.18)`
- **Subtle Borders (`--border-subtle`)**: `rgba(255, 255, 255, 0.08)`
- **Text Primary (`--text-primary`)**: `#F1F5F9` (Crisp off-white)
- **Text Secondary (`--text-secondary`)**: `#94A3B8` (Muted cool slate)
- **Text Muted (`--text-muted`)**: `#64748B`

#### Glassmorphism & Shadow Depth
- **Floating Utility Shadow**: `box-shadow: 0 16px 40px rgba(16, 117, 143, 0.07), 0 4px 12px rgba(0, 0, 0, 0.03);`
- **Card Soft Shadow**: `box-shadow: 0 10px 30px rgba(16, 117, 143, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);`
- **Hover Lift Shadow**: `box-shadow: 0 20px 48px rgba(16, 117, 143, 0.12); transform: translateY(-3px);`
- **Backdrop Filters**: `backdrop-filter: blur(16px) saturate(160%);`

---

### Corner 2: Card Geometry & Corner Radii

- **Hero & Primary Section Containers**: `border-radius: 32px` (Smooth, oversized modern squircle)
- **Feature, Diagnostic & Department Cards**: `border-radius: 20px` to `24px`
- **Form Groups, Metric Insets & Image Containers**: `border-radius: 16px`
- **Interactive Action Buttons & Badges**: Sleek **Pill-Shape** (`border-radius: 9999px`) or `border-radius: 12px`
- **Whitespace Channels**: Distinct content zones are partitioned by generous `48px` to `72px` vertical margins rather than heavy, rigid dividers.

---

### Corner 3: Typography & Visual Hierarchy

- **Typeface Scale**: Plus Jakarta Sans / Inter, system-ui fallback.
- **Structural Headlines**:
  - **H1 (Hero Titles)**: `font-size: 2.75rem` (44px), `font-weight: 800`, `letter-spacing: -0.025em`, `line-height: 1.15`.
  - **H2 (Section Titles)**: `font-size: 1.85rem` (30px), `font-weight: 700`, `letter-spacing: -0.02em`.
  - **H3 (Card Titles)**: `font-size: 1.15rem` (18px), `font-weight: 700`.
- **Two-Tone Value Headlines**:
  - Structure: Primary slate font paired with highlighted teal value words.
  - *Example*: `"Better Health, [Brighter Future]"` where `[Brighter Future]` is highlighted in `var(--accent)`.
  - *Example*: `"Quantum Precision, [Clinical Confidence]"`
- **Body & Captions**:
  - Body: `font-size: 0.95rem`, `font-weight: 400`, `line-height: 1.6`.
  - Micro-metadata: `font-size: 0.78rem`, `font-weight: 600`, uppercase tracking `0.04em`.

---

### Corner 4: Layout Grid, Padding & Whitespace

- **8-Point Spatial Grid**: All margins, paddings, gaps adhere to multiples of 8 (`8px`, `16px`, `24px`, `32px`, `48px`, `64px`).
- **Internal Card Breathing Room**: Minimum `padding: 24px` (`p-6`) on all feature and diagnostic cards.
- **Touch Target Integrity**: Minimum `44px × 44px` clickable touch targets for all buttons, inputs, and toggles.
- **Floating Over-The-Fold Utility Bar**:
  - Sits centered, overlapping the hero fold by `-32px`.
  - Includes input icon, select dropdown, QPU hardware switch, and high-contrast pill CTA.

---

## 3. Component Architecture

1. **GlobalNav**:
   - Translucent glass bar (`backdrop-filter: blur(16px)`).
   - Healthcare cross + Quantum logo.
   - Active pill navigation states.
   - Interactive Light/Dark theme toggle.
   - Live SQLite history drawer launcher and hardware status beacon.
2. **Hero Workspace**:
   - Clean, rounded container (`border-radius: 32px`).
   - Two-tone editorial title.
   - 3 bullet trust pills (Standards, QPU Hardware, Grad-CAM).
3. **Floating Diagnostic Launcher**:
   - Quick-select module (Breast Cancer, Heart QML, Alzheimer's).
   - Hardware selector (Statevector Simulator vs IQM Garnet 20-Qubit).
   - "Launch Diagnostic Suite →" pill CTA.
4. **Disease Department Cards**:
   - Oversized squircle cards with custom teal icon containers.
   - Detailed tags and clear CTA links.
5. **Teal Enterprise Healthcare Banner**:
   - Deep teal gradient container with 4 key metrics and outline icons.
6. **Recent Runs Table & History Drawer**:
   - Styled with clean white/obsidian cards, pill badges for outcomes, and hardware execution details.
