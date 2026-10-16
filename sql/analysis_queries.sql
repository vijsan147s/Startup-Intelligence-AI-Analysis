-- Startup Intelligence AI — SQL Analysis Queries
-- Database: SQLite (data/processed/startup_funding.db)
-- Table: funding_events
-- All queries assume the cleaned dataset has been loaded.

-- 1. Overall funding-event counts
SELECT COUNT(*) AS total_events FROM funding_events;

-- 2. Events with valid amounts
SELECT COUNT(amount_clean) AS events_with_valid_amount FROM funding_events;

-- 3. Total recorded funding (USD)
SELECT SUM(amount_clean) AS total_recorded_funding_usd
FROM funding_events
WHERE amount_clean IS NOT NULL;

-- 4. Annual funding-event trends
SELECT
    year,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd,
    AVG(amount_clean) AS avg_funding_usd,
    COUNT(amount_clean) AS events_with_amount
FROM funding_events
WHERE year IS NOT NULL
GROUP BY year
ORDER BY year;

-- 5. Sector-level funding analysis
SELECT
    industry_clean,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd,
    AVG(amount_clean) AS avg_funding_usd,
    COUNT(amount_clean) AS events_with_amount
FROM funding_events
GROUP BY industry_clean
ORDER BY total_funding_usd DESC;

-- 6. City-level funding analysis
SELECT
    city_clean,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd,
    AVG(amount_clean) AS avg_funding_usd,
    COUNT(amount_clean) AS events_with_amount
FROM funding_events
GROUP BY city_clean
ORDER BY total_funding_usd DESC;

-- 7. Funding stage/type distribution
SELECT
    investment_type_clean,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd,
    COUNT(amount_clean) AS events_with_amount
FROM funding_events
GROUP BY investment_type_clean
ORDER BY event_count DESC;

-- 8. Top investors by appearance frequency
SELECT
    investors_name,
    COUNT(*) AS appearance_count
FROM funding_events
WHERE investors_name IS NOT NULL AND investors_name != ''
GROUP BY investors_name
ORDER BY appearance_count DESC
LIMIT 20;

-- 9. Sector-by-year comparison
SELECT
    industry_clean,
    year,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd
FROM funding_events
WHERE year IS NOT NULL
GROUP BY industry_clean, year
ORDER BY industry_clean, year;

-- 10. City-by-year comparison
SELECT
    city_clean,
    year,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd
FROM funding_events
WHERE year IS NOT NULL
GROUP BY city_clean, year
ORDER BY city_clean, year;

-- 11. Data-quality summary
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN amount_clean IS NULL THEN 1 ELSE 0 END) AS missing_amounts,
    SUM(CASE WHEN date_parsed IS NULL THEN 1 ELSE 0 END) AS missing_dates,
    SUM(CASE WHEN industry_clean = 'Unknown' THEN 1 ELSE 0 END) AS unknown_industry,
    SUM(CASE WHEN city_clean = 'Unknown' THEN 1 ELSE 0 END) AS unknown_city
FROM funding_events;

-- 12. Monthly funding trends
SELECT
    year,
    month,
    COUNT(*) AS event_count,
    SUM(amount_clean) AS total_funding_usd
FROM funding_events
WHERE year IS NOT NULL AND month IS NOT NULL
GROUP BY year, month
ORDER BY year, month;

-- 13. Funding amount distribution (quartiles)
SELECT
    MIN(amount_clean) AS min_amount,
    MAX(amount_clean) AS max_amount,
    AVG(amount_clean) AS mean_amount,
    COUNT(amount_clean) AS valid_count
FROM funding_events
WHERE amount_clean IS NOT NULL;
