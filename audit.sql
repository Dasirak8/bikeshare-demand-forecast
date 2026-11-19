-- Independent check of the simplest forecast in the case study.
-- Load the pinned day.csv into daily (dteday TEXT, cnt INTEGER).
-- Apply the test date filter AFTER computing the rolling mean so that
-- the first test day still has access to the preceding week.
WITH forecasts AS (
    SELECT
        dteday,
        cnt AS actual,
        AVG(cnt) OVER (
            ORDER BY dteday ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS prior_week_mean,
        LAG(cnt, 1) OVER (ORDER BY dteday) AS yesterday
    FROM daily
)
SELECT
    COUNT(*) AS test_days,
    AVG(ABS(actual - prior_week_mean)) AS prior_week_mean_mae,
    AVG(ABS(actual - yesterday)) AS yesterday_mae
FROM forecasts
WHERE dteday BETWEEN '2012-09-03' AND '2012-12-31';
