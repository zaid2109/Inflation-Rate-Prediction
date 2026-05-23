const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, TableOfContents,
  PageBreak
} = require('docx');
const fs = require('fs');

const BLUE = "1a56db";
const DARK = "111827";
const MID = "374151";
const LIGHT = "6B7280";
const ACCENT = "dc2626";
const GREEN = "059669";
const PURPLE = "7c3aed";
const ORANGE = "d97706";
const BG_BLUE = "EFF6FF";
const BG_GREEN = "ECFDF5";
const BG_RED = "FEF2F2";
const BG_PURPLE = "F5F3FF";
const BG_ORANGE = "FFFBEB";
const BG_GRAY = "F9FAFB";
const BORDER_COLOR = "E5E7EB";

const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function p(text, color = DARK, size = 22, bold = false, italic = false) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, color, size, bold, italics: italic, font: "Calibri" })]
  });
}

function spacer(size = 120) {
  return new Paragraph({ spacing: { after: size }, children: [] });
}

function sectionTitle(text, color = BLUE) {
  return new Paragraph({
    spacing: { before: 320, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: color } },
    children: [new TextRun({ text: text.toUpperCase(), color, size: 24, bold: true, font: "Calibri" })]
  });
}

function bullet(text, level = 0, color = DARK, bold = false) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: [new TextRun({ text, color, size: 21, bold, font: "Calibri" })]
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { after: 80 },
    children: [new TextRun({ text, color: DARK, size: 21, font: "Calibri" })]
  });
}

function callout(text, bgColor = BG_BLUE, borderColor = BLUE) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [200, 9160],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: { ...noBorders, left: { style: BorderStyle.SINGLE, size: 12, color: borderColor } },
            width: { size: 200, type: WidthType.DXA },
            shading: { fill: bgColor, type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 60, right: 60 },
            children: [new Paragraph({ children: [] })]
          }),
          new TableCell({
            borders: { ...noBorders, right: { style: BorderStyle.NONE, size: 0, color: bgColor } },
            width: { size: 9160, type: WidthType.DXA },
            shading: { fill: bgColor, type: ShadingType.CLEAR },
            margins: { top: 100, bottom: 100, left: 160, right: 160 },
            children: [new Paragraph({ children: [new TextRun({ text, color: DARK, size: 21, font: "Calibri", italics: true })] })]
          })
        ]
      })
    ]
  });
}

function twoColTable(rows, col1Width = 2800, col2Width = 6560, headerBg = BLUE) {
  const totalWidth = 9360;
  const tableRows = rows.map((row, i) => {
    const isHeader = i === 0;
    const bg = isHeader ? headerBg : (i % 2 === 0 ? "FFFFFF" : BG_GRAY);
    const textColor = isHeader ? "FFFFFF" : DARK;
    return new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        width: { size: ci === 0 ? col1Width : col2Width, type: WidthType.DXA },
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          children: [new TextRun({ text: cell, color: textColor, size: isHeader ? 20 : 20, bold: isHeader, font: "Calibri" })]
        })]
      }))
    });
  });
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: [col1Width, col2Width], rows: tableRows });
}

function threeColTable(rows, widths = [2200, 3580, 3580], headerBg = BLUE) {
  const tableRows = rows.map((row, i) => {
    const isHeader = i === 0;
    const bg = isHeader ? headerBg : (i % 2 === 0 ? "FFFFFF" : BG_GRAY);
    const textColor = isHeader ? "FFFFFF" : DARK;
    return new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        width: { size: widths[ci], type: WidthType.DXA },
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          children: [new TextRun({ text: cell, color: textColor, size: 20, bold: isHeader, font: "Calibri" })]
        })]
      }))
    });
  });
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: widths, rows: tableRows });
}

function fourColTable(rows, widths = [2000, 2000, 2680, 2680], headerBg = BLUE) {
  const tableRows = rows.map((row, i) => {
    const isHeader = i === 0;
    const bg = isHeader ? headerBg : (i % 2 === 0 ? "FFFFFF" : BG_GRAY);
    const textColor = isHeader ? "FFFFFF" : DARK;
    return new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        width: { size: widths[ci], type: WidthType.DXA },
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          children: [new TextRun({ text: cell, color: textColor, size: 20, bold: isHeader, font: "Calibri" })]
        })]
      }))
    });
  });
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: widths, rows: tableRows });
}

function fiveColTable(rows, widths = [1600, 1800, 1920, 1920, 2120], headerBg = BLUE) {
  const tableRows = rows.map((row, i) => {
    const isHeader = i === 0;
    const bg = isHeader ? headerBg : (i % 2 === 0 ? "FFFFFF" : BG_GRAY);
    const textColor = isHeader ? "FFFFFF" : DARK;
    return new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        width: { size: widths[ci], type: WidthType.DXA },
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          children: [new TextRun({ text: cell, color: textColor, size: 19, bold: isHeader, font: "Calibri" })]
        })]
      }))
    });
  });
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: widths, rows: tableRows });
}

function phaseBox(phaseNum, title, weeks, color, tasks) {
  const rows = [
    new TableRow({
      children: [
        new TableCell({
          borders,
          columnSpan: 2,
          width: { size: 9360, type: WidthType.DXA },
          shading: { fill: color, type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 160, right: 160 },
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: `PHASE ${phaseNum}  `, color: "FFFFFF", size: 20, bold: true, font: "Calibri" }),
                new TextRun({ text: `${title}`, color: "FFFFFF", size: 22, bold: true, font: "Calibri" }),
                new TextRun({ text: `   ●  ${weeks}`, color: "FFFFFF", size: 18, font: "Calibri" }),
              ]
            })
          ]
        })
      ]
    }),
    ...tasks.map((task, i) => new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 400, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? "FFFFFF" : BG_GRAY, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 160, right: 80 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${i + 1}.`, color: color, size: 20, bold: true, font: "Calibri" })] })]
        }),
        new TableCell({
          borders,
          width: { size: 8960, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? "FFFFFF" : BG_GRAY, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: task, color: DARK, size: 20, font: "Calibri" })] })]
        })
      ]
    }))
  ];
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [400, 8960], rows });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ]
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22, color: DARK } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Calibri", color: DARK },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Calibri", color: BLUE },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Calibri", color: MID },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Table({
            width: { size: 10080, type: WidthType.DXA },
            columnWidths: [5040, 5040],
            rows: [new TableRow({
              children: [
                new TableCell({
                  borders: { ...noBorders, bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE } },
                  width: { size: 5040, type: WidthType.DXA },
                  shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
                  margins: { top: 40, bottom: 60, left: 0, right: 0 },
                  children: [new Paragraph({ children: [new TextRun({ text: "INFLATION RATE PREDICTION SYSTEM", color: BLUE, size: 18, bold: true, font: "Calibri" })] })]
                }),
                new TableCell({
                  borders: { ...noBorders, bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE } },
                  width: { size: 5040, type: WidthType.DXA },
                  shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
                  margins: { top: 40, bottom: 60, left: 0, right: 0 },
                  children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Complete Development Plan  |  College Capstone  |  2026", color: LIGHT, size: 17, font: "Calibri" })] })]
                })
              ]
            })]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR } },
            spacing: { before: 80 },
            children: [
              new TextRun({ text: "Inflation Rate Prediction System  |  Confidential College Project Document  |  Page ", color: LIGHT, size: 17, font: "Calibri" }),
              new TextRun({ children: [PageNumber.CURRENT], color: LIGHT, size: 17, font: "Calibri" }),
              new TextRun({ text: " of ", color: LIGHT, size: 17, font: "Calibri" }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], color: LIGHT, size: 17, font: "Calibri" }),
            ]
          })
        ]
      })
    },
    children: [

      // COVER PAGE
      new Table({
        width: { size: 10080, type: WidthType.DXA },
        columnWidths: [10080],
        rows: [new TableRow({
          children: [new TableCell({
            borders: noBorders,
            width: { size: 10080, type: WidthType.DXA },
            shading: { fill: "0f172a", type: ShadingType.CLEAR },
            margins: { top: 800, bottom: 800, left: 600, right: 600 },
            children: [
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "INFLATION RATE PREDICTION SYSTEM", color: "60a5fa", size: 20, bold: true, font: "Calibri" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Complete Development Plan", color: "FFFFFF", size: 52, bold: true, font: "Calibri" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "The definitive guide to building a top-class ML forecasting system", color: "94a3b8", size: 22, font: "Calibri", italics: true })] }),
              spacer(200),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "College Capstone Project  |  Machine Learning & Data Science  |  2026", color: "64748b", size: 19, font: "Calibri" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Version 2.0  |  Complete Edition — All Layers Covered", color: "64748b", size: 19, font: "Calibri" })] }),
            ]
          })]
        })]
      }),
      spacer(200),

      // Stats row
      new Table({
        width: { size: 10080, type: WidthType.DXA },
        columnWidths: [2016, 2016, 2016, 2016, 2016],
        rows: [new TableRow({
          children: [
            ["6 Phases", "16+ Weeks", "4 ML Models", "20+ Features", "Full Stack"],
          ][0].map((text, i) => new TableCell({
            borders: { ...noBorders, bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE } },
            width: { size: 2016, type: WidthType.DXA },
            shading: { fill: BG_BLUE, type: ShadingType.CLEAR },
            margins: { top: 120, bottom: 120, left: 80, right: 80 },
            children: [
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, color: BLUE, size: 22, bold: true, font: "Calibri" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: ["End-to-End", "Walk-Forward CV", "SHAP Explainability", "REST API + Docker", "Dashboard + Docs"][i], color: LIGHT, size: 17, font: "Calibri" })] })
            ]
          }))
        })]
      }),
      spacer(300),

      new Paragraph({ children: [new PageBreak()] }),

      // TABLE OF CONTENTS
      h("Table of Contents"),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 1
      h("1. Executive Summary & Project Vision"),
      p("This document is the complete, authoritative development plan for the Inflation Rate Prediction System — a college capstone project that demonstrates end-to-end machine learning engineering from raw data ingestion to a deployed, interactive dashboard. Every layer of the system is specified here: data, modeling, engineering, evaluation, presentation, and academic output.", DARK, 22),
      spacer(),
      p("What separates this project from a typical college submission is depth at every layer. Most projects train one model, display a chart, and call it done. This plan covers four models with proper temporal validation, SHAP explainability, ensemble stacking, uncertainty quantification, regime-aware modeling, a production-grade API, Docker deployment, and a research-quality academic report with literature citations and ablation studies.", DARK, 22),
      spacer(),
      callout("The single goal: when an examiner reviews this project, every technical decision should have a clear rationale, every result should be reproducible, and the system should feel like something built by someone who understands real-world forecasting — not just someone who completed a tutorial.", BG_BLUE, BLUE),
      spacer(200),

      h("2. Project Objectives", HeadingLevel.HEADING_2),
      bullet("Build a machine-learning system that accurately forecasts the Consumer Price Index (CPI) and derived inflation rate using macroeconomic time-series data from 1990–2025."),
      bullet("Train, compare, and ensemble four models: Linear Regression baseline, ARIMA/SARIMA, XGBoost with SHAP, and LSTM with uncertainty quantification."),
      bullet("Implement walk-forward (expanding-window) validation — the correct evaluation methodology for time-series — rather than a naive 80/20 split."),
      bullet("Expose the best model via a FastAPI REST endpoint with Docker containerisation, proper input validation, and Redis caching."),
      bullet("Build an interactive React/Streamlit dashboard showing historical vs. predicted inflation, model comparison, feature importance, and a scenario explorer."),
      bullet("Produce a 10–15 page academic report with literature review, ablation study, and comparison against professional forecasters."),
      bullet("Detect and handle structural breaks (COVID-19 regime shift) explicitly rather than treating the full series as homogeneous."),
      spacer(200),

      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 2
      h("3. Technology Stack"),
      p("Every technology choice below has a specific reason. Do not swap libraries arbitrarily — the plan is designed so that each tool integrates cleanly with the others.", LIGHT, 20, false, true),
      spacer(),

      fiveColTable([
        ["Layer", "Technology", "Version", "Purpose", "Why This Choice"],
        ["Language", "Python", "3.11+", "Core development", "Best ML ecosystem; type hints for API"],
        ["Data Ingestion", "pandas-datareader", "0.10+", "Pull FRED, World Bank data", "Official FRED API wrapper"],
        ["Data Ingestion", "fredapi", "0.5+", "FRED macro data", "Direct API, more control than datareader"],
        ["Data Ingestion", "yfinance", "0.2+", "Oil price, equity indices", "Free, reliable financial data"],
        ["Data Ingestion", "pytrends", "4.9+", "Google Trends alternative data", "Unique signal; no competitor uses this"],
        ["Data Processing", "pandas", "2.0+", "Cleaning, feature engineering", "Standard; excellent time-series support"],
        ["Data Processing", "NumPy", "1.24+", "Numerical operations", "Required by all ML libs"],
        ["Data Quality", "great_expectations", "0.17+", "Data validation schema", "Professional-grade, shows rigor"],
        ["ML / Stats", "scikit-learn", "1.3+", "Ridge/Lasso regression, CV", "Baseline + preprocessing pipelines"],
        ["ML / Stats", "statsmodels", "0.14+", "ARIMA, SARIMA, Diebold-Mariano", "Statistical testing built-in"],
        ["ML / Stats", "pmdarima", "2.0+", "auto_arima parameter selection", "Avoids manual grid search for ARIMA"],
        ["Ensemble", "XGBoost", "2.0+", "Gradient boosting regressor", "Non-linear; best single model benchmark"],
        ["Explainability", "shap", "0.43+", "SHAP feature importance", "Industry standard; required for top marks"],
        ["Deep Learning", "TensorFlow / Keras", "2.14+", "LSTM sequence model", "Stable; good Keras API for LSTM"],
        ["Uncertainty", "MAPIE", "0.6+", "Conformal prediction intervals", "Statistically valid uncertainty bounds"],
        ["MLOps", "MLflow", "2.8+", "Experiment tracking, model registry", "Industry standard; shows production awareness"],
        ["MLOps", "Evidently AI", "0.4+", "Data & model drift detection", "Automated monitoring; rare in college projects"],
        ["API", "FastAPI", "0.104+", "REST prediction endpoint", "Async; auto Swagger docs; type-safe"],
        ["API", "Pydantic", "2.0+", "Input/output validation schemas", "Built into FastAPI; prevents bad inputs"],
        ["Caching", "Redis", "4.0+", "Cache repeated predictions", "Shows production engineering mindset"],
        ["Containerisation", "Docker + Compose", "24+", "Reproducible deployment", "One-command spin-up; examiner-proof"],
        ["Dashboard", "Streamlit OR React", "1.28+ / 18+", "Interactive visualisation", "Streamlit: fast; React: more impressive"],
        ["Charting", "Plotly", "5.17+", "Interactive charts", "Better than matplotlib for web dashboards"],
        ["Version Control", "Git + GitHub", "—", "Source control, CI/CD", "Tag v1.0.0; show clean commit history"],
        ["CI/CD", "GitHub Actions", "—", "Automated test runs on push", "Proves engineering discipline"],
        ["Testing", "pytest", "7.4+", "Unit + integration tests", "Required for any serious project"],
        ["Environment", "Anaconda / venv", "—", "Reproducible dev environment", "requirements.txt + environment.yml"],
        ["Notebooks", "Jupyter Lab", "4.0+", "EDA and model training docs", "Annotated notebooks = clear narrative"],
      ], [1800, 2000, 1480, 2080, 2000]),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 3
      h("4. Data Plan"),
      h("4.1 Primary Data Sources", HeadingLevel.HEADING_2),
      p("Use at least 35 years of monthly data (1990–2025) across all sources. Quarterly data (GDP) must be forward-filled to monthly frequency. Document every source, frequency, and access method in your data README.", DARK, 22),
      spacer(),

      fiveColTable([
        ["Dataset", "Source", "Frequency", "Years", "FRED Series ID"],
        ["CPI (All Urban, NSA)", "FRED / BLS", "Monthly", "1990–2025", "CPIAUCNS"],
        ["CPI (Seasonally Adj.)", "FRED / BLS", "Monthly", "1990–2025", "CPIAUCSL"],
        ["Core CPI (ex food/energy)", "FRED / BLS", "Monthly", "1990–2025", "CPILFESL"],
        ["PCE Price Index", "FRED", "Monthly", "1990–2025", "PCEPI"],
        ["GDP Growth Rate", "World Bank / FRED", "Quarterly", "1990–2025", "A191RL1Q225SBEA"],
        ["M2 Money Supply", "FRED", "Monthly", "1990–2025", "M2SL"],
        ["M2 Velocity", "FRED", "Quarterly", "1990–2025", "M2V"],
        ["Unemployment Rate", "BLS / FRED", "Monthly", "1990–2025", "UNRATE"],
        ["Fed Funds Rate", "FRED", "Monthly", "1990–2025", "FEDFUNDS"],
        ["10-Year Treasury Yield", "FRED", "Monthly", "1990–2025", "GS10"],
        ["2-Year Treasury Yield", "FRED", "Monthly", "1990–2025", "GS2"],
        ["Yield Curve (10Y-2Y)", "Computed", "Monthly", "1990–2025", "T10Y2Y"],
        ["Crude Oil Price (WTI)", "EIA / yfinance", "Monthly", "1990–2025", "DCOILWTICO"],
        ["PPI (All commodities)", "BLS / FRED", "Monthly", "1990–2025", "PPIACO"],
        ["Housing Starts", "FRED", "Monthly", "1990–2025", "HOUST"],
        ["Retail Sales", "FRED", "Monthly", "1990–2025", "RSXFS"],
        ["Consumer Sentiment (UMich)", "FRED", "Monthly", "1990–2025", "UMCSENT"],
        ["Baltic Dry Index", "yfinance (BDI)", "Monthly", "2000–2025", "^BDI"],
        ["US Dollar Index (DXY)", "yfinance", "Monthly", "1990–2025", "DX-Y.NYB"],
        ["Google Trends: inflation", "pytrends", "Monthly", "2004–2025", "Custom keyword"],
        ["Google Trends: price increase", "pytrends", "Monthly", "2004–2025", "Custom keyword"],
      ], [2400, 1800, 1200, 1200, 2760]),
      spacer(160),
      callout("Why 20+ features? Real forecasting models use diverse signals. More signals = more predictive power, but also more engineering discipline needed (proper train/test isolation, multicollinearity checks, feature selection). Showing you navigated this complexity is what earns top marks.", BG_ORANGE, ORANGE),
      spacer(200),

      h("4.2 Feature Engineering", HeadingLevel.HEADING_2),
      p("Feature engineering is where domain knowledge turns raw data into model signal. Document every transformation decision — why you created each feature, what economic logic supports it.", DARK, 22),
      spacer(),
      h("Lag Features", HeadingLevel.HEADING_3),
      bullet("CPI(t-1), CPI(t-3), CPI(t-6), CPI(t-12) — inflation is autocorrelated; yesterday predicts tomorrow"),
      bullet("Fed Funds Rate lagged 3 and 6 months — monetary policy takes time to affect prices"),
      bullet("Oil price lagged 1 and 3 months — energy costs propagate through supply chain with a delay"),
      bullet("M2 money supply lagged 6 and 12 months — money supply effects on inflation are famously delayed"),
      spacer(80),
      h("Rolling Statistics", HeadingLevel.HEADING_3),
      bullet("3-month and 12-month rolling mean of CPI — trend smoothing"),
      bullet("3-month and 12-month rolling standard deviation of CPI — volatility signal"),
      bullet("12-month rolling mean of unemployment — labour market trend"),
      bullet("6-month rolling mean of oil price — energy trend"),
      spacer(80),
      h("Rate-of-Change Features", HeadingLevel.HEADING_3),
      bullet("Month-over-month % change for CPI, M2, unemployment, oil"),
      bullet("Year-over-year % change for all primary indicators"),
      bullet("Acceleration: second-order difference of CPI (is inflation speeding up or slowing down?)"),
      spacer(80),
      h("Interaction & Composite Features", HeadingLevel.HEADING_3),
      bullet("M2 growth rate × interest rate differential — classic monetarist interaction term"),
      bullet("Yield curve spread (10Y minus 2Y Treasury) — well-documented inflation/recession predictor"),
      bullet("Real interest rate = Fed Funds Rate minus CPI — negative real rates historically predict rising inflation"),
      bullet("Output gap proxy = GDP growth deviation from rolling 5-year average"),
      spacer(80),
      h("Temporal & Regime Features", HeadingLevel.HEADING_3),
      bullet("Month-of-year dummy variables (1–12) — captures seasonal patterns in CPI"),
      bullet("COVID regime binary flag: 1 for March 2020 onwards, 0 before — handles structural break"),
      bullet("Post-GFC flag: 1 for 2010–2019 — captures the persistently low-inflation decade"),
      bullet("Hidden Markov Model regime indicator: automatically detected high/low inflation regime (see Section 5.4)"),
      spacer(80),
      h("Alternative Data Features", HeadingLevel.HEADING_3),
      bullet("Google Trends index for 'inflation' search volume — consumer anxiety precedes CPI"),
      bullet("Google Trends index for 'price increase' — business sentiment signal"),
      bullet("Baltic Dry Index — global shipping costs predict goods inflation 2–3 months ahead"),
      spacer(160),
      callout("Data Leakage Warning: Never use any target-future information in features. All lags must use strictly past values. Use TimeSeriesSplit for all cross-validation — never shuffle temporal data. This is the single most common mistake in student time-series projects.", BG_RED, ACCENT),
      spacer(80),

      h("4.3 Data Quality Pipeline", HeadingLevel.HEADING_2),
      p("Most projects do df.fillna() and move on. A professional data pipeline documents every quality decision. This is rare in college projects and will be noticed.", DARK, 22),
      spacer(),
      bullet("Outlier detection: flag values beyond 3 standard deviations using z-score; log them; do not silently drop"),
      bullet("Missing value strategy: forward-fill for macro series with < 5% missing; interpolation for GDP quarterly-to-monthly alignment; document every imputation choice"),
      bullet("Great Expectations schema: define explicit data expectations (CPI > 0, unemployment between 0–20%, etc.); run validation on every new data pull"),
      bullet("Alignment check: verify all series share the same date index after merge; log any gaps"),
      bullet("Stationarity tests: run Augmented Dickey-Fuller test on CPI and log results; apply differencing if needed for ARIMA"),
      bullet("Correlation heatmap: check for multicollinearity (r > 0.95) between features; document which features were dropped and why"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 4
      h("5. Model Architecture — All Four + Ensemble"),
      p("Every model must be logged to MLflow with full hyperparameters, metrics, and the serialised artefact. Never delete a run — the progression of experiments is part of the academic record.", DARK, 22),
      spacer(160),

      h("5.1 Baseline — Ridge/Lasso Linear Regression", HeadingLevel.HEADING_2),
      bullet("Library: scikit-learn RidgeCV / LassoCV"),
      bullet("Purpose: establishes an interpretable benchmark; all complex models must beat this to justify their complexity"),
      bullet("Preprocessing pipeline: StandardScaler → feature selector → Ridge (use sklearn Pipeline so scaler is never fitted on test data)"),
      bullet("Hyperparameter: alpha selected via cross-validation using TimeSeriesSplit(n_splits=5)"),
      bullet("Output: coefficient plot showing which features the model relies on — this is your first feature importance signal"),
      bullet("Regularisation note: use Lasso to drive some coefficients to zero — effectively built-in feature selection"),
      spacer(160),

      h("5.2 Statistical — ARIMA / SARIMA", HeadingLevel.HEADING_2),
      bullet("Library: statsmodels + pmdarima (auto_arima)"),
      bullet("Purpose: captures autocorrelation and seasonality in the univariate CPI series; pure time-series benchmark"),
      bullet("Parameter selection: use auto_arima with stepwise=True, information_criterion='aic' — log the selected (p,d,q)(P,D,Q,s) order"),
      bullet("Seasonal model: fit SARIMA(p,d,q)(P,D,Q,12) with monthly seasonality period"),
      bullet("Diagnostic checks: Ljung-Box test on residuals; QQ plot; ACF/PACF of residuals — document all in the academic report"),
      bullet("Limitation: univariate only; cannot benefit from exogenous variables (use SARIMAX as an extension if time permits)"),
      spacer(160),

      h("5.3 Ensemble — XGBoost Regressor + SHAP", HeadingLevel.HEADING_2),
      bullet("Library: xgboost + shap"),
      bullet("Purpose: captures non-linear relationships between macroeconomic features; typically the best single model"),
      bullet("Pipeline: ColumnTransformer for scaling + XGBRegressor in a single sklearn Pipeline"),
      bullet("Hyperparameter tuning: GridSearchCV with TimeSeriesSplit(n_splits=5); tune: n_estimators, max_depth, learning_rate, subsample, colsample_bytree"),
      bullet("SHAP explainability: compute SHAP values for the test set; plot summary plot (beeswarm), bar plot (mean absolute SHAP), and force plot for specific months"),
      bullet("Partial dependence: plot PDP for top-3 features (oil price, M2 growth, yield curve spread) to show non-linear relationships"),
      bullet("Feature importance: export top-10 features with SHAP values — include this chart in the dashboard and the academic report"),
      spacer(160),

      h("5.4 Deep Learning — LSTM with Uncertainty", HeadingLevel.HEADING_2),
      bullet("Library: TensorFlow 2.x / Keras + MAPIE"),
      bullet("Purpose: sequence modelling with 24-month lookback window; captures long-range temporal dependencies"),
      bullet("Architecture: InputLayer → LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1)"),
      bullet("Training: Adam optimiser; MSE loss; EarlyStopping(patience=15, monitor='val_loss'); ReduceLROnPlateau"),
      bullet("Data prep: MinMaxScaler fitted only on training data; inverse-transform all predictions before evaluation"),
      bullet("Uncertainty quantification: wrap trained LSTM in MAPIE MapieRegressor with conformal prediction; produces 90% prediction intervals (not just point estimates)"),
      bullet("This is the difference between 'I built an LSTM' and 'I built an LSTM with statistically valid uncertainty bounds'"),
      spacer(160),

      h("5.5 Advanced: Ensemble Stacking (Top-Class Addition)", HeadingLevel.HEADING_2),
      callout("This is the feature that will put your project above every other submission. Almost no college project attempts this. Stacking combines the predictions of your four models as input features to a meta-learner, producing a final prediction that is better than any individual model.", BG_GREEN, GREEN),
      spacer(80),
      bullet("Level-0 (base) models: XGBoost, LSTM, Ridge, ARIMA — each produces out-of-fold predictions on training data"),
      bullet("Level-1 (meta) model: Ridge regression or simple neural network trained on the out-of-fold predictions"),
      bullet("Implementation: use sklearn StackingRegressor or custom cross-val-predict loop with TimeSeriesSplit"),
      bullet("Expected improvement: 5–15% reduction in RMSE vs. best single model in most literature"),
      bullet("Report this as your main result: 'The stacked ensemble achieved RMSE of X vs. RMSE of Y for the best single model'"),
      spacer(160),

      h("5.6 Regime-Aware Modeling (Top-Class Addition)", HeadingLevel.HEADING_2),
      bullet("Library: hmmlearn (Hidden Markov Model)"),
      bullet("Fit a 2-state HMM on CPI series to automatically identify high-inflation vs. low-inflation regimes"),
      bullet("Use the detected regime as an additional binary feature in all models"),
      bullet("Alternative: train separate XGBoost models for pre-2020 and post-2020 data; compare performance on 2021–2025 test set"),
      bullet("Visualise regime periods on the historical chart: shaded background for detected high-inflation periods"),
      bullet("Discuss in the academic report: how did regime awareness change feature importance?"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 5
      h("6. Evaluation Framework"),
      h("6.1 Core Metrics", HeadingLevel.HEADING_2),

      fiveColTable([
        ["Metric", "Formula", "Target", "Why It Matters", "Where to Report"],
        ["MAE", "Mean Absolute Error", "< 0.3 pp", "Interpretable: avg error in % pts", "All tables + dashboard"],
        ["RMSE", "Root Mean Squared Error", "< 0.5 pp", "Penalises large errors more", "All tables + dashboard"],
        ["MAPE", "Mean Abs Percentage Error", "< 5%", "Scale-independent comparison", "Academic report"],
        ["R²", "Coefficient of Determination", "> 0.90", "Variance explained", "Model comparison table"],
        ["Directional Accuracy", "% correct up/down calls", "> 70%", "Practical decision-making value", "Dashboard KPI card"],
        ["Diebold-Mariano", "Statistical significance test", "p < 0.05", "Proves best model is not just lucky", "Academic report"],
      ]),
      spacer(160),

      h("6.2 Walk-Forward (Expanding Window) Validation", HeadingLevel.HEADING_2),
      p("This is the most important methodological upgrade in this plan. A simple 80/20 split is not sufficient for time-series evaluation.", DARK, 22),
      spacer(),
      bullet("Split the data into an initial training window (1990–2010) and a test period (2011–2025)"),
      bullet("At each step: train on all data up to month t, predict month t+1, advance by 1 month"),
      bullet("This produces 168+ out-of-sample predictions (2011–2025) that simulate real deployment conditions"),
      bullet("Compute metrics only on the walk-forward predictions — this is what matters"),
      bullet("Use sklearn TimeSeriesSplit(n_splits=10) for hyperparameter tuning within the training window"),
      bullet("Document in the report: why walk-forward validation gives a more honest assessment than cross-validation on shuffled data"),
      spacer(160),

      h("6.3 Ablation Study (Academic Requirement)", HeadingLevel.HEADING_2),
      p("An ablation study removes one component at a time and measures the impact on performance. This is standard in ML research and almost never seen in college projects.", DARK, 22),
      spacer(),

      fiveColTable([
        ["Experiment", "What Is Removed", "Expected RMSE", "Metric Change", "Conclusion"],
        ["Full model (baseline)", "Nothing removed", "~0.35", "—", "Full feature set"],
        ["No lag features", "CPI(t-1) to CPI(t-12)", "~0.52", "+49%", "Lags are critical"],
        ["No alternative data", "Google Trends, BDI", "~0.38", "+9%", "Alt data helps ~9%"],
        ["No interaction terms", "M2 × rate, yield curve", "~0.42", "+20%", "Interactions matter"],
        ["No regime feature", "COVID dummy, HMM state", "~0.47", "+34%", "Regime critical post-2020"],
        ["No rolling statistics", "3m/12m mean, std dev", "~0.40", "+14%", "Rolling stats add value"],
        ["Univariate only", "All exogenous features", "~0.58", "+66%", "Multivariate >> univariate"],
      ]),
      spacer(80),
      callout("Present this table in your academic report and your presentation. It demonstrates scientific rigor and proves you understand which parts of your system are actually doing the work.", BG_GREEN, GREEN),
      spacer(160),

      h("6.4 Comparison Against Professional Forecasters", HeadingLevel.HEADING_2),
      bullet("Download the Survey of Professional Forecasters (SPF) consensus CPI forecast from the Philadelphia Fed"),
      bullet("Download the Cleveland Fed's inflation nowcast (available publicly)"),
      bullet("Compare your model's RMSE and MAE against both on the overlapping test period"),
      bullet("Even if your model does not beat them, the comparison is academically valuable and shows awareness of the field"),
      bullet("If your stacked ensemble beats the SPF consensus on any sub-period — this is your headline result"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 6
      h("7. API Development — FastAPI + Docker"),
      h("7.1 Endpoints", HeadingLevel.HEADING_2),

      fiveColTable([
        ["Method", "Endpoint", "Input", "Output", "Description"],
        ["POST", "/predict", "JSON feature payload", "Inflation forecast + CI", "Primary prediction endpoint"],
        ["POST", "/predict/batch", "Array of feature payloads", "Array of forecasts", "Batch predictions"],
        ["GET", "/models", "—", "Model list + metrics", "Available model versions"],
        ["GET", "/history", "?months=24", "Actual vs predicted", "Last N months comparison"],
        ["GET", "/features", "—", "Feature importance", "SHAP values for latest prediction"],
        ["GET", "/health", "—", "Service status", "Uptime, model version, data date"],
        ["GET", "/docs", "—", "Swagger UI", "Auto-generated by FastAPI"],
      ]),
      spacer(160),

      h("7.2 Input Validation Schema", HeadingLevel.HEADING_2),
      p("Every input must be validated before reaching the model. This prevents crashes during the live demo and shows professional engineering.", DARK, 22),
      spacer(),
      bullet("Pydantic BaseModel for all request/response schemas with type annotations and field validators"),
      bullet("Validate ranges: unemployment 0–30%, oil price 0–500, interest rate -5% to 25%"),
      bullet("Handle missing fields: return 422 Unprocessable Entity with clear error message, not a Python traceback"),
      bullet("Model versioning: accept optional model_version parameter to switch between Ridge, ARIMA, XGBoost, LSTM, Ensemble"),
      bullet("Response always includes: point_estimate, lower_bound, upper_bound, confidence_level, model_used, prediction_date"),
      spacer(160),

      h("7.3 Caching & Performance", HeadingLevel.HEADING_2),
      bullet("Redis cache for repeated identical requests (TTL: 1 hour) — prevents redundant model inference"),
      bullet("Async endpoint handlers using FastAPI's async/await — handles concurrent requests without blocking"),
      bullet("Model loaded at startup (not per-request) — startup time once vs. latency on every request"),
      bullet("Response time target: < 100ms for cached, < 500ms for fresh LSTM inference"),
      spacer(160),

      h("7.4 Docker Setup", HeadingLevel.HEADING_2),
      bullet("docker-compose.yml with three services: api (FastAPI), redis (caching), mlflow (experiment tracking)"),
      bullet("Single command to run the entire system: docker compose up — examiner can verify the demo in one step"),
      bullet("Multi-stage Dockerfile: builder stage installs dependencies; runtime stage is smaller and cleaner"),
      bullet("Environment variables via .env file: FRED_API_KEY, MODEL_PATH, REDIS_URL — never hardcode secrets"),
      bullet("Volume mounts for /models and /data/raw so trained models persist across container restarts"),
      bullet("Add a DEMO.md with exact commands: git clone → docker compose up → open browser"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 7
      h("8. Dashboard — Complete Specification"),
      h("8.1 Section: At a Glance (Hero)", HeadingLevel.HEADING_2),
      bullet("Large display of current inflation rate (e.g., 3.0%) with colour coding: green < 2%, amber 2–4%, red > 4%"),
      bullet("Status badge: 'At Target' / 'Elevated' / 'High' with plain-English explanation"),
      bullet("Delta indicator: up/down arrow with change vs. last month"),
      bullet("'What does this mean for me?' plain-language card with grocery/rent/mortgage examples"),
      spacer(160),

      h("8.2 Section: Key Insights (Three Cards)", HeadingLevel.HEADING_2),
      bullet("Card 1 — Trend: is inflation going up, down, or stable? Based on rolling 3-month change"),
      bullet("Card 2 — Model Forecast: what does the model expect next month? Include confidence interval"),
      bullet("Card 3 — Biggest Driver: which feature has the highest SHAP value right now? Dynamic, not hardcoded"),
      spacer(160),

      h("8.3 Section: Historical Chart", HeadingLevel.HEADING_2),
      bullet("Interactive Plotly line chart of CPI YoY% from 1990 to present"),
      bullet("Colour-coded background bands: green (1–3% healthy), amber (3–6% elevated), red (> 6% high)"),
      bullet("Fed 2% target as horizontal dashed reference line"),
      bullet("Confidence interval shaded band on the prediction portion (not just a single line)"),
      bullet("10 Years / 25 Years / All Time toggle buttons"),
      bullet("Hover tooltip showing exact value, date, and status for any point"),
      bullet("Regime markers: vertical shaded bands for COVID period and GFC period with labels"),
      spacer(160),

      h("8.4 Section: Model Comparison Scorecard", HeadingLevel.HEADING_2),
      callout("This section is the most important academic addition to the dashboard. It makes your ML work visible to the examiner without them reading the code.", BG_BLUE, BLUE),
      spacer(80),
      bullet("Table showing MAE, RMSE, MAPE, R², Directional Accuracy for all four models + ensemble"),
      bullet("Best model highlighted with a trophy icon and coloured border"),
      bullet("Bar chart comparing RMSE across models — visual reinforcement of the table"),
      bullet("Model selector dropdown: choose which model's predictions appear on the historical chart"),
      spacer(160),

      h("8.5 Section: SHAP Feature Importance", HeadingLevel.HEADING_2),
      bullet("Horizontal bar chart of top-10 features ranked by mean absolute SHAP value"),
      bullet("Bar colour: blue for positive impact on inflation, red for negative impact"),
      bullet("Time slider: show how feature importance has shifted over time (e.g., oil price dominated 2021–2022)"),
      bullet("Click on any feature to see its partial dependence plot — how changes in that variable affect the prediction"),
      spacer(160),

      h("8.6 Section: Scenario Explorer", HeadingLevel.HEADING_2),
      bullet("Sliders for: unemployment rate, federal interest rate, oil price, M2 growth rate, consumer sentiment"),
      bullet("Real-time prediction update as sliders move (debounced API call to /predict)"),
      bullet("Comparison panel: current conditions vs. user-defined scenario side by side"),
      bullet("Narrative insight: dynamically generated from SHAP values"),
      spacer(160),

      h("8.7 Section: API Response Preview", HeadingLevel.HEADING_2),
      bullet("A small code block showing the live JSON response from the /predict endpoint"),
      bullet("Automatically updates when the scenario explorer changes"),
      bullet("Syntax-highlighted JSON display — proves the backend is real and working"),
      spacer(160),

      h("8.8 Section: Data Pipeline Status", HeadingLevel.HEADING_2),
      bullet("Table showing each data source with last-updated timestamp"),
      bullet("Green tick / red cross status indicators"),
      bullet("Shows examiners the system is connected to real data, not hardcoded demo values"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 8
      h("9. Development Phases — 16-Week Timeline"),
      spacer(80),

      phaseBox(1, "Project Setup & Data Acquisition", "Weeks 1–2", "1a56db", [
        "Create GitHub repository with branch strategy: main / dev / feature-* / experiment-*",
        "Set up virtual environment; create requirements.txt and environment.yml",
        "Register FRED API key; write data ingestion scripts with error handling and rate-limit backoff",
        "Download and persist all raw datasets as CSV in /data/raw/ with download timestamps",
        "Set up MLflow tracking server (local); configure experiment names",
        "Write unit tests for all ingestion functions; target 80%+ test coverage",
        "Set up great_expectations suite; define data expectations for each source",
        "Configure GitHub Actions CI: run tests on every push to dev branch"
      ]),
      spacer(160),

      phaseBox(2, "EDA & Feature Engineering", "Weeks 3–4", "7c3aed", [
        "Exploratory data analysis: distribution plots, correlation heatmap, ACF/PACF for CPI series",
        "Stationarity analysis: ADF test on all series; log results and apply differencing where needed",
        "Handle missing values using documented strategy (forward-fill, interpolation, flag-and-impute)",
        "Align time indices: convert quarterly GDP to monthly by forward-fill; verify all series share same index",
        "Engineer all lag, rolling, interaction, and regime features (see Section 4.2)",
        "Multicollinearity check: VIF analysis; drop features with VIF > 10 after discussion",
        "Train-test split: initial training window 1990–2010; test period 2011–2025 (strict temporal isolation)",
        "Run great_expectations validation on processed feature matrix; fix any violations",
        "Export processed feature matrix to /data/processed/ with schema documentation"
      ]),
      spacer(160),

      phaseBox(3, "Model Training & Tuning", "Weeks 5–8", "059669", [
        "Train Baseline Ridge/Lasso; log to MLflow; generate coefficient importance plot",
        "Fit ARIMA/SARIMA with auto_arima; log (p,d,q)(P,D,Q,s) order; run diagnostic tests",
        "Train XGBoost with 5-fold TimeSeriesSplit walk-forward CV; tune with GridSearchCV",
        "Compute SHAP values for XGBoost; generate summary plot, bar plot, force plots",
        "Build LSTM pipeline: MinMaxScaler → windowed sequences (24-month lookback) → 2-layer LSTM",
        "Add conformal prediction via MAPIE to LSTM; verify 90% coverage on test data",
        "Implement ensemble stacking: out-of-fold predictions from base models → meta-learner",
        "Fit Hidden Markov Model for regime detection; add regime feature to all models",
        "Run full walk-forward evaluation on 2011–2025; compute all metrics",
        "Diebold-Mariano test: compare best model vs. baseline; confirm statistical significance",
        "Run ablation study; fill in results table from Section 6.3",
        "Compare against SPF and Cleveland Fed forecasts on overlapping period",
        "Log all experiments to MLflow; select best model and register it in MLflow Model Registry"
      ]),
      spacer(160),

      phaseBox(4, "API Development", "Weeks 9–10", "d97706", [
        "Design all endpoints: POST /predict, POST /predict/batch, GET /models, GET /history, GET /features, GET /health",
        "Build Pydantic schemas for all request and response models with field validators",
        "Serialise best model with joblib / TF SavedModel; load at API startup",
        "Implement Redis caching for /predict endpoint with 1-hour TTL",
        "Add async request handling; benchmark response time (target: < 500ms for LSTM)",
        "Write pytest integration tests for all endpoints; test edge cases",
        "Write Dockerfile and docker-compose.yml (API + Redis + MLflow services)",
        "Test one-command spin-up: docker compose up; verify all services start correctly",
        "Document all endpoints with examples in README.md and Postman collection"
      ]),
      spacer(160),

      phaseBox(5, "Dashboard & Visualisation", "Weeks 11–12", "dc2626", [
        "Build all 8 sections from Section 8 of this document",
        "Implement interactive Plotly chart with confidence bands, regime markers, and time range toggle",
        "Build model comparison scorecard table and RMSE comparison bar chart",
        "Build SHAP feature importance bar chart with partial dependence plots",
        "Build scenario explorer with sliders connected to live API /predict endpoint",
        "Add API response preview JSON block that updates with scenario explorer",
        "Connect dashboard to FastAPI backend via HTTP requests; handle loading states and errors",
        "Test on different screen sizes; ensure all charts are readable"
      ]),
      spacer(160),

      phaseBox(6, "Testing, Documentation & Submission", "Weeks 13–16", "374151", [
        "End-to-end integration testing: run full pipeline from raw data ingestion to dashboard display",
        "Edge-case testing: missing inputs, future dates, extreme slider values, network failures",
        "Write comprehensive README.md with architecture diagram, setup instructions, and sample outputs",
        "Annotate all four Jupyter notebooks: EDA, feature engineering, modelling, evaluation",
        "Write academic college report (10–15 pages): all sections from Section 11.1",
        "Include minimum 8 academic citations (see Section 11 for recommendations)",
        "Record 5-minute demo video: data pipeline → MLflow runs → API call → dashboard live",
        "Tag final release v1.0.0 on GitHub with clean commit history",
        "Submit all artefacts: GitHub repo, notebooks, report PDF, demo video"
      ]),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 9
      h("10. Engineering & MLOps"),
      h("10.1 Folder Structure", HeadingLevel.HEADING_2),

      twoColTable([
        ["Path", "Contents & Purpose"],
        ["/data/raw/", "Original downloaded CSVs with download timestamps. Never modify."],
        ["/data/processed/", "Feature-engineered datasets. Versioned by date. Include schema docs."],
        ["/notebooks/01_eda.ipynb", "EDA: distributions, correlations, ACF/PACF, stationarity tests."],
        ["/notebooks/02_feature_eng.ipynb", "Feature engineering: all transformations with economic rationale."],
        ["/notebooks/03_models.ipynb", "Model training: all four models, hyperparameter tuning, SHAP analysis."],
        ["/notebooks/04_evaluation.ipynb", "Walk-forward evaluation: all metrics, ablation study, professional comparison."],
        ["/src/ingest.py", "Data downloading functions: one function per source, error handling, caching."],
        ["/src/features.py", "Feature engineering pipeline: sklearn-compatible transformer classes."],
        ["/src/train.py", "Model training and saving: consistent interface for all four models."],
        ["/src/evaluate.py", "Metrics computation: MAE, RMSE, MAPE, R², Directional Accuracy, Diebold-Mariano."],
        ["/src/predict.py", "Inference utilities: load model, preprocess input, run prediction, format output."],
        ["/src/regime.py", "Hidden Markov Model regime detection; exports regime feature series."],
        ["/api/main.py", "FastAPI application: all endpoints, startup event, exception handlers."],
        ["/api/schemas.py", "Pydantic request/response models for all endpoints."],
        ["/dashboard/app.py", "Streamlit or React dashboard: all 8 sections from Section 8."],
        ["/models/", "Serialised model files: best_model.pkl, lstm_model.keras, meta_learner.pkl."],
        ["/tests/", "Pytest test suite: unit tests for src/, integration tests for api/."],
        ["/docker-compose.yml", "Three-service Docker setup: api, redis, mlflow."],
        ["/Dockerfile", "Multi-stage build for the FastAPI service."],
        ["/.github/workflows/", "GitHub Actions CI: run tests on push; lint with flake8."],
        ["/mlruns/", "MLflow experiment logs — commit this to GitHub as part of the record."],
        ["/requirements.txt", "All Python dependencies with pinned versions."],
        ["/README.md", "Architecture diagram, setup instructions, demo instructions, sample outputs."],
      ], 2800, 6560),
      spacer(160),

      h("10.2 MLOps — Model Lifecycle", HeadingLevel.HEADING_2),
      bullet("Every training run logged to MLflow: hyperparameters, all metrics, model artefact, and environment"),
      bullet("Use MLflow Model Registry to track model versions: Staging → Production → Archived"),
      bullet("Evidently AI: generate a data drift report comparing training distribution to 2024–2025 data; include in academic report"),
      bullet("Model drift detection: if walk-forward RMSE on the last 12 months exceeds 2× the overall test RMSE, flag a retraining trigger"),
      bullet("Reproducibility: set random seeds everywhere; log Python + library versions in MLflow; document hardware"),
      spacer(160),

      h("10.3 Testing Strategy", HeadingLevel.HEADING_2),
      bullet("Unit tests: every function in /src/ has at least one test; test edge cases (empty dataframe, NaN inputs, zero variance)"),
      bullet("Integration tests: API endpoints tested with TestClient; verify correct status codes and response schemas"),
      bullet("Data pipeline tests: run great_expectations suite on /data/processed/ before every model training"),
      bullet("Temporal integrity tests: assert that no test-set dates appear in the training feature matrix"),
      bullet("Target: 80%+ test coverage (measure with pytest-cov; include coverage badge in README)"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 10
      h("11. Academic Report — Structure & Requirements"),
      p("The academic report is your primary evaluated artefact. It must read like a research paper, not a project log. Every technical decision must have a rationale cited either from literature or from your own experimental results.", DARK, 22),
      spacer(160),

      h("11.1 Report Structure (10–15 Pages)", HeadingLevel.HEADING_2),

      twoColTable([
        ["Section", "Required Content"],
        ["Abstract (200 words)", "Problem, approach, key result (best model RMSE), conclusion. Write this last."],
        ["1. Introduction", "Why inflation forecasting matters; what this project contributes; structure of the paper."],
        ["2. Literature Review", "5–8 cited papers; summarise what prior work found; identify the gap this project addresses."],
        ["3. Data & Features", "All 20+ features with economic rationale; data quality decisions; stationarity analysis."],
        ["4. Methodology", "Walk-forward validation explanation; all four model architectures; ensemble stacking design."],
        ["5. Results", "Model comparison table; walk-forward RMSE plot over time; SHAP feature importance chart."],
        ["6. Ablation Study", "Table from Section 6.3; discuss which components contributed most."],
        ["7. Comparison vs. Professionals", "Your model vs. SPF consensus and Cleveland Fed nowcast."],
        ["8. Limitations", "What the model cannot do; when it would fail; future enhancements."],
        ["9. Conclusion", "Summary of findings; practical implications; what you would do differently."],
        ["References", "Minimum 8 academic or institutional citations in consistent format (APA or IEEE)."],
        ["Appendix A", "Full model hyperparameter tables; MLflow run IDs for reproducibility."],
        ["Appendix B", "Full data source list with URLs, access dates, and license information."],
      ], 2400, 6960),
      spacer(160),

      h("11.2 Recommended Literature to Cite", HeadingLevel.HEADING_2),
      bullet("Medeiros, M.C. et al. (2021) — 'Forecasting Inflation in a Data-Rich Environment' — directly relevant"),
      bullet("Stock, J.H. and Watson, M.W. (2007) — 'Why Has US Inflation Become Harder to Forecast?' — classic reference"),
      bullet("Coulombe, P.G. et al. (2020) — 'How is Machine Learning Useful for Macroeconomic Forecasting?' — ML vs. ARIMA"),
      bullet("Diebold, F.X. and Mariano, R.S. (1995) — 'Comparing Predictive Accuracy' — the test you will use; always cite"),
      bullet("Lundberg, S.M. and Lee, S.I. (2017) — 'A Unified Approach to Interpreting Model Predictions' — the SHAP paper"),
      bullet("Hochreiter, S. and Schmidhuber, J. (1997) — 'Long Short-Term Memory' — the original LSTM paper"),
      bullet("Cleveland Fed Inflation Nowcasting — institutional source; cite when comparing against professionals"),
      bullet("Survey of Professional Forecasters — Federal Reserve Bank of Philadelphia; cite for consensus comparison"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 11
      h("12. Risks & Mitigations"),

      fiveColTable([
        ["Risk", "Likelihood", "Impact", "Mitigation", "Owner"],
        ["API rate limit on FRED", "Medium", "Low", "Cache raw data locally on first pull; use batch download; implement exponential backoff", "Phase 1"],
        ["Data leakage in CV", "High", "Critical", "Use TimeSeriesSplit exclusively; never shuffle; write a temporal integrity test", "Phase 2"],
        ["LSTM overfitting", "High", "Medium", "Dropout layers; early stopping; validate on 2019–2025; use MAPIE intervals", "Phase 3"],
        ["Structural break (COVID)", "High", "High", "COVID regime dummy; train pre/post-2020 models separately; compare results", "Phase 3"],
        ["Conformal prediction coverage failure", "Medium", "Low", "Verify empirically that 90% CI contains 90% of test actuals; document if it does not", "Phase 3"],
        ["Docker networking issues", "Medium", "Medium", "Test docker compose up on a clean machine before submission; provide fallback local run instructions", "Phase 4"],
        ["Scope creep", "High", "Medium", "Freeze MVP feature list by end of Week 4; use GitHub Issues to track extras separately", "All phases"],
        ["Demo failure during presentation", "Low", "High", "Prepare offline demo with pre-recorded video as backup; test on presentation machine", "Phase 6"],
        ["pytrends API changes", "Medium", "Low", "Google Trends data is bonus; all core analysis works without it", "Phase 1"],
        ["SHAP computation too slow for LSTM", "Medium", "Low", "Use KernelExplainer with a sample of 200 background points; or use XGBoost SHAP only", "Phase 3"],
      ], [1800, 1000, 800, 3400, 1360]),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 12
      h("13. Final Deliverables Checklist"),
      p("Every item below must be complete before submission. Check them off in your GitHub README with screenshots as evidence.", DARK, 22),
      spacer(160),

      h("13.1 Code & Engineering", HeadingLevel.HEADING_2),
      bullet("GitHub repository: public, clean commit history, descriptive commit messages, tagged v1.0.0"),
      bullet("CI/CD: GitHub Actions running tests on every push; badge visible in README"),
      bullet("All four model training scripts with MLflow logging"),
      bullet("Ensemble stacking implementation with walk-forward evaluation"),
      bullet("FastAPI service with all six endpoints, input validation, and Redis caching"),
      bullet("docker-compose.yml: single-command spin-up verified on a clean machine"),
      bullet("Test suite: 80%+ coverage; all tests passing in CI"),
      bullet("requirements.txt and environment.yml with pinned versions"),
      spacer(160),

      h("13.2 Notebooks", HeadingLevel.HEADING_2),
      bullet("01_eda.ipynb: annotated with markdown cells explaining every finding; ADF tests documented"),
      bullet("02_feature_eng.ipynb: every feature documented with economic rationale"),
      bullet("03_models.ipynb: all four models trained; SHAP analysis; ensemble stacking; MLflow run IDs shown"),
      bullet("04_evaluation.ipynb: walk-forward results; ablation study table; comparison vs. professionals"),
      spacer(160),

      h("13.3 Dashboard", HeadingLevel.HEADING_2),
      bullet("All 8 sections from Section 8 implemented and functional"),
      bullet("Model comparison scorecard with all four models + ensemble"),
      bullet("Live SHAP feature importance chart"),
      bullet("Scenario explorer connected to live API"),
      bullet("Confidence interval bands on the historical chart"),
      bullet("Regime markers on the historical chart"),
      bullet("API response JSON preview block"),
      spacer(160),

      h("13.4 Academic Deliverables", HeadingLevel.HEADING_2),
      bullet("College report (PDF): 10–15 pages, all sections from Section 11.1, minimum 8 citations"),
      bullet("Ablation study table included in report"),
      bullet("Comparison vs. SPF and Cleveland Fed included in report"),
      bullet("Limitations section: honest, specific, and linked to your experimental results"),
      bullet("Demo video: 5 minutes; covers data pipeline, MLflow, API call, dashboard live interaction"),
      bullet("README.md: architecture diagram, one-command setup, sample API request/response, screenshots"),
      spacer(200),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 13
      h("14. Presentation Strategy — What Top Projects Do"),
      h("14.1 The Live Demo Checklist", HeadingLevel.HEADING_2),
      bullet("Never show screenshots of a working system — show the system actually working"),
      bullet("Open a terminal: run docker compose up live; let the examiner see it start"),
      bullet("Open Postman or curl: make a live /predict API call; show the JSON response"),
      bullet("Open the dashboard: adjust the scenario sliders; show the prediction updating in real time"),
      bullet("Open MLflow: show the experiment list; click on your best run; show the metrics and artefact"),
      bullet("Have a backup: pre-recorded 5-minute video in case of network or hardware failure"),
      spacer(160),

      h("14.2 The One Killer Insight", HeadingLevel.HEADING_2),
      p("Every memorable presentation has one genuine discovery that the presenter found in the data. Here are candidates to look for:", DARK, 22),
      spacer(),
      bullet("Did the yield curve spread (10Y-2Y) invert before the 2021–2022 inflation spike? Plot it and discuss."),
      bullet("Does Google Trends for 'inflation' lead CPI by 1–2 months? Run a cross-correlation and show the lag."),
      bullet("Did your regime-aware model perform significantly better on 2020–2022 than the non-regime model? Show the numbers."),
      bullet("Which features dominated SHAP importance in 2022 vs. 2024? Has the inflation driver shifted from energy to services?"),
      bullet("Did your stacked ensemble beat the SPF consensus on any 12-month window? If yes, this is your headline."),
      spacer(160),

      h("14.3 Handling Examiner Questions", HeadingLevel.HEADING_2),
      bullet("'Why not use a Transformer/attention model?' — LSTM is appropriate for 35 years of monthly data (~420 points); Transformers need more data and add complexity without guaranteed benefit at this scale."),
      bullet("'Why not use real-time data?' — Out of scope by design; FRED data updates monthly which is appropriate for monthly CPI forecasting."),
      bullet("'How do you know your model is not overfitting?' — Walk-forward validation; the 168 test predictions were never seen during training. Show the RMSE plot over time."),
      bullet("'What would you do with more time?' — Transformer model; real-time FRED data streaming; multi-country comparison; production cloud deployment with auto-retraining."),
      spacer(200),

      // FINAL CALLOUT
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [9360],
        rows: [new TableRow({
          children: [new TableCell({
            borders: { ...noBorders, left: { style: BorderStyle.SINGLE, size: 16, color: GREEN } },
            width: { size: 9360, type: WidthType.DXA },
            shading: { fill: BG_GREEN, type: ShadingType.CLEAR },
            margins: { top: 200, bottom: 200, left: 300, right: 300 },
            children: [
              new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "What separates a top-class project from a good one", color: GREEN, size: 26, bold: true, font: "Calibri" })] }),
              new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "It is not the number of features or the complexity of the model. It is intellectual honesty and depth of understanding demonstrated at every layer:", color: DARK, size: 21, font: "Calibri" })] }),
              new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Every technical decision has a documented rationale. Every result is reproducible. Limitations are stated clearly and linked to evidence. The ablation study proves which parts of the system actually matter. The live demo works without rehearsal. And somewhere in the analysis, there is one genuine discovery — something you found in the data that surprised you.", color: MID, size: 21, font: "Calibri", italics: true })] }),
              new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: "Follow this plan completely and that project will be yours. Good luck.", color: GREEN, size: 21, bold: true, font: "Calibri" })] }),
            ]
          })]
        })]
      }),
      spacer(200),

    ]
  }]
});

const outputPath = 'Inflation_Prediction_Complete_Plan.docx';
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log('Document generated: ' + outputPath + ' (' + Math.round(buffer.length / 1024) + ' KB)');
}).catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
