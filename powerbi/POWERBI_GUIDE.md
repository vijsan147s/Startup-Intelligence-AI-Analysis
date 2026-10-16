# Power BI Dashboard — Detailed Build Guide

Companion to the **Startup Intelligence AI** project.
Dataset: [`startup_funding_powerbi.xlsx`](startup_funding_powerbi.xlsx) — 3,044 events, Jan 2015–Jan 2020.

Time needed: ~30–45 minutes in **Power BI Desktop** (free download from Microsoft).

---

## Part A — Load and prepare the data (10 min)

### A1. Import the Excel file
1. Open Power BI Desktop → close the splash screen if shown.
2. **Home ribbon → Get Data → Excel**.
3. Browse to `powerbi/startup_funding_powerbi.xlsx` → **Open**.
4. In the Navigator, tick the checkboxes for all 5 sheets:
   `Events`, `Yearly`, `Sector`, `City`, `FundingType`.
5. Click **Load** (not Transform). Wait for "3,044 rows" to finish loading.

### A2. Verify data types (important — wrong types break visuals)
1. Switch to **Data view** (table icon on the left).
2. Select the **Events** table (Fields pane → click `Events`).
3. Click each column header and check the type shown in **Column tools**:
   - `Date` → **Date** (if it shows Text: Column tools → Data type → Date)
   - `Year`, `Month`, `Quarter` → **Whole Number**
   - `AmountUSD` → **Decimal Number**
   - `Startup`, `Industry`, `SubVertical`, `City`, `FundingType`, `Investors` → **Text**
4. Rename columns if needed by double-clicking the header (keep the names from the file).

### A3. (Optional) Hide helper tables
- In **Model view**, you will see 5 separate tables with no relationships — that is fine.
- Build every visual from the **Events** table so slicers filter everything.
- The `Yearly`/`Sector`/`City`/`FundingType` sheets are reference tables for cross-checking your chart totals.

### A4. Save your work
- **File → Save As** → `powerbi/Startup_Intelligence_Dashboard.pbix` (same folder as this guide).

---

## Part B — Create DAX measures (10 min)

Measures are reusable calculations. Create them all on the **Events** table:

1. In **Report view**, right-click the **Events** table in the Fields pane → **New measure**.
2. Delete the placeholder text and paste one formula at a time from below, then press **Enter**.
3. After creating each measure, set its format (select measure → **Measure tools** ribbon):
   - Money measures → Format: **Currency ($, English-US)**, 0–2 decimals.
   - `Missing Amount %` → Format: **Percentage**, 1 decimal.
   - Counts → Format: **Whole number**.

```dax
-- 1. Total recorded funding
Total Funding USD = SUM ( Events[AmountUSD] )

-- 2. Number of funding events
Event Count = COUNTROWS ( Events )

-- 3. Events that have a funding amount
Events With Amount = COUNT ( Events[AmountUSD] )

-- 4. Typical (median) funding — robust to outliers
Median Funding = MEDIAN ( Events[AmountUSD] )

-- 5. Average funding (skewed by mega-rounds — compare with median)
Average Funding = AVERAGE ( Events[AmountUSD] )

-- 6. Total in billions, for card titles
Total Funding B = DIVIDE ( [Total Funding USD], 1000000000 )

-- 7. Share of events missing an amount
Missing Amount % =
DIVIDE (
    [Event Count] - [Events With Amount],
    [Event Count]
)

-- 8. Largest single round in the current filter context
Largest Round = MAX ( Events[AmountUSD] )
```

Quick check: drag `Event Count` onto a blank canvas as a **Card** — it must show **3044**.
If it shows blank, your Events table did not load correctly — repeat Part A.

---

## Part C — Build the pages (20 min)

General technique for every visual below: click a blank area of the canvas →
click the visual icon → drag fields from the Fields pane into the visual's field wells.
Resize by dragging corners. Rename visual titles by selecting the visual →
**Format (paint roller) → General → Title → On → type the title**.

> Add this text box to EVERY page (Insert → Text box):
> "Historical data: Jan 2015 – Jan 2020. Recorded totals only — not current market data, not investment advice."

### Page 1 — "Overview" (rename Page 1: right-click tab → Rename)
1. **4 Cards** (Visualizations → Card):
   - Card 1 field: `Event Count` → title "Funding Events"
   - Card 2 field: `Total Funding B` → title "Recorded Funding (B USD)"
   - Card 3 field: `Median Funding` → title "Median Round (USD)"
   - Card 4 field: `Missing Amount %` → title "Events Missing Amount"
   - Layout: arrange in one row across the top.
2. **Line chart** (Line chart icon):
   - X-axis: `Year` · Y-axis: `Event Count` → title "Funding Events per Year"
3. **Clustered bar chart**:
   - Y-axis: `Industry` · X-axis: `Total Funding USD`
   - Filters pane (with visual selected) → Top N: Top **5** by Total Funding USD
   - Title: "Top 5 Industries by Recorded Funding"
4. **3 Slicers** (Slicer icon), one each for `Year`, `Industry`, `City`.
   - Select a slicer → Format → Slicer settings → Style: **Tile** (nicer for demos).
5. Test: click `Bengaluru` in the City slicer — all 4 cards and both charts must change.

### Page 2 — "Industry" (Home → Insert → New page, then rename)
1. **Bar chart:** Y-axis `Industry`, X-axis `Total Funding USD`, Top 10 filter → "Top 10 Industries by Funding".
2. **Bar chart:** Y-axis `Industry`, X-axis `Event Count`, Top 10 filter → "Top 10 Industries by Event Count".
3. **Table** (Table icon): columns `Industry`, `Event Count`, `Total Funding USD`, `Median Funding`, `Average Funding`.
   - Click the `Total Funding USD` column header in the table → sort descending (arrow icon).
4. **Slicer:** `Year` (to compare sectors across years).
5. Talking point for viva: Consumer Internet has the most *events* (941) but E-Commerce has the highest *total* ($8.2B) — counts vs totals tell different stories.

### Page 3 — "City" (new page → rename)
1. **Map:** use the **Azure Map** visual (or Map if offline):
   - Location: `City` · Bubble size: `Total Funding USD` · Title: "Funding by City".
   - Note: a few non-Indian cities exist in the data (e.g. Menlo Park); that is correct, they are in the source file.
2. **Bar chart:** `City` → `Event Count`, Top 10 → "Top 10 Cities by Events".
3. **Table:** `City`, `Event Count`, `Total Funding USD`, `Median Funding`.
4. Add a text box: "City names normalized: Bangalore→Bengaluru, Gurgaon→Gurugram (see project docs)."

### Page 4 — "Investors" (new page → rename)
1. **Bar chart:** Y-axis `Investors`, X-axis `Event Count`, Top 15.
   - Filters pane → exclude blanks: drag `Investors` to Filters → uncheck "(Blank)".
2. **Table:** `Investors`, `Event Count`, `Total Funding USD`.
3. Add a text box (required for correctness):
   "Frequency = appearances in the dataset. It does NOT imply investor quality or returns."

---

## Part D — Polish like a pro (5 min)

1. **Theme:** View ribbon → Themes → pick one (e.g. "Executive") and keep it on all pages.
2. **Numbers:** select each money visual → Format → Data labels → Display units: **Millions/Billions**, 1 decimal.
3. **Sort sanity:** bar charts sort by value descending (⋯ menu on visual → Sort by → measure).
4. **Q&A visual (great for demos):** on Overview, Insert → Q&A → try typing
   "total funding by city" or "events by year". If a question misreads, use
   Q&A → "Teach Q&A" to fix the phrasing once.
5. **Drill-through (optional):** right-click Industry bar chart → Drill through → add a
   hidden "Details" page with a table of Startup/Date/AmountUSD for that industry.

---

## Part E — Share / submit

| Goal | Steps |
|------|-------|
| Class submission | File → Export → **Export to PDF** (all pages) → submit the PDF |
| Slides | File → Export → **Export to PowerPoint** (live-data slides) |
| Online link | **Publish** button (needs free PowerBI.com account) → File → Embed → share link |
| Keep repo small | Save the `.pbix` next to this guide; do NOT commit multi-100 MB `.pbix` files to git |

## Part F — Troubleshooting

| Symptom | Fix |
|---------|-----|
| Card shows blank | Measure created on wrong table — recreate under **Events** |
| Slicer doesn't filter a chart | Chart was built from a summary sheet (Sector/City) instead of **Events** — rebuild from Events |
| Map shows no bubbles | Location field must be `City` (text); check data type is Text, not Number |
| Dates show as text | Data view → select `Date` column → Data type → **Date** |
| Totals don't match README ($38.06B) | A slicer is active — clear selections (eraser icon on each slicer) |
| Excel changed — how to refresh? | Regenerate the xlsx, then in Power BI: **Home → Refresh** (same file path required) |
