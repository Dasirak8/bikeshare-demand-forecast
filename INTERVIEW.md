# Interview walkthrough

## The 45-second version

> The project asked whether a bike-share operator could use information available tonight to predict tomorrow’s total trips. It compared three simple rules with Ridge regression and a shallow decision tree on two years of Capital Bikeshare data. The split kept dates in order, so each model predicted days that came after its training period. The average of the previous seven days won validation: it was off by about 764 trips a day. But in the final 120 days it missed by about 978, while simply repeating yesterday’s count missed by about 919. I wouldn’t claim the rolling average is better in production. I’d keep both as benchmarks and test them on newer months using archived weather forecasts or known service notices.

This was built with AI assistance. Before presenting it, rerun the project and explain the parts you actually inspected or changed; don't claim you wrote every line or deployed it.

## Walk through the six charts

| Chart | What to say |
|---|---|
| Demand history | Each point is one day’s total system rentals. The 28-day line smooths fluctuations for display. It is not the model. |
| Training timeline | Train on earlier dates, test predictions on the next block, then expand the training period. The orange block never chooses the winner. |
| Mean validation error | Lower is better. The 7-day average beat the other four candidates here without fitting parameters. |
| Fold errors | Ridge did better in the last two folds. No method won every block. |
| Actual vs forecast | The rolling average tracks the broad demand level but smooths away abrupt changes. On October 29, 2012, it expected 7,047 trips and recorded usage was 22. |
| Signed error | Bars above zero are overpredictions; bars below zero are underpredictions. Look for persistent bias or large misses, not just the average. |

## Questions to be ready for

**What is the simplest forecast, exactly?** If the seven completed daily counts add up to 28,000 trips, tomorrow’s forecast is 4,000. Today’s completed count is included; tomorrow’s actual count is not.

**What was actually trained?** Ridge learned weights on eight past and calendar inputs. It scaled those inputs using the training rows only. The tree learned decision splits from those same inputs, with depth capped at four. The winning average has no coefficients to fit, only seven observed counts to update.

**Why not do a random split?** The real job is forecasting the future using the past. A random split could train on dates that come after dates used as tests and give a misleading result.

**Why was yesterday’s actual available for a test date?** This is a series of nightly one-day forecasts. Each forecast uses the counts that would already have been observed by then, including earlier days in the final test. It isn't one forecast made 120 days ahead.

**What's leakage here?** Tomorrow’s observed weather and tomorrow’s casual/registered counts would be unavailable tonight. The rider counts also add up to the target, so including them would largely give away the answer.

**What does MAE mean?** Average the size of the misses without their signs. If you overpredict by 200 and underpredict by 400, MAE is 300 trips per day. Lower is better.

**Why not switch the winner after seeing the final test?** You can report yesterday did better on that period, but promoting it on the same evidence would turn the holdout into another selection step. Use a fresh period to confirm any change.

**Can this tell you where to move bikes?** No. This predicts completed trips across the entire system. It has no station-level inventory, empty-dock or lost-demand data.

## One thing to practice

Run `python run.py`, then open `outputs/test_predictions.csv`. For October 29, explain the difference between the forecast, actual, and error in your own words before looking back at the report.
