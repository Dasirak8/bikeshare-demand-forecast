"""Run the case study and regenerate the figures, PDF and result tables."""
import json
import sqlite3
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

from forecast import ROOT, evaluate, load_snapshot, make_features

OUT = ROOT / 'outputs'
INK, MUTED, TEAL, BLUE, ORANGE = '#142A3A', '#445E70', '#007F78', '#4277A1', '#A74921'
LIGHT, GRID = '#E4F2F0', '#DCE6EB'


def page(number, topic, title):
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.text(.097, .948, f'BIKESHARE / FORECASTING CASE STUDY     {number:02d} / {topic.upper()}',
             size=9, weight='bold', color=TEAL)
    fig.text(.095, .881, title, size=31, weight='bold', color=INK, va='top', linespacing=1.1)
    fig.add_artist(Line2D([.095, .9], [.059, .059], transform=fig.transFigure, color=GRID, lw=1))
    fig.text(.095, .038, 'CAPITAL BIKESHARE  /  HISTORICAL BACKTEST', size=8, color=MUTED)
    fig.text(.9, .038, f'{number} / 5', size=8, color=MUTED, ha='right')
    return fig


def note(fig, y, title, text, width=93):
    fig.text(.097, y, title, size=11, color=TEAL, weight='bold', va='top')
    fig.text(.097, y-.028, textwrap.fill(text, width=width), size=11, color=INK,
             va='top', linespacing=1.48)


def chart(name, size, draw):
    fig, ax = plt.subplots(figsize=size, layout='constrained', facecolor='white')
    draw(ax)
    path = OUT / 'figures' / f'{name}.svg'
    fig.savefig(path, bbox_inches='tight', pad_inches=.12, metadata={'Date': None})
    # Matplotlib pads path lines with spaces; strip them for clean Git diffs.
    svg = path.read_text(encoding='utf-8')
    path.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


def make_report(daily, result):
    test = result['predictions']
    mean_mae = result['test_mae']
    ref_mae = result['baseline_mae']
    gap = (mean_mae / ref_mae - 1) * 100
    worst_day = test.error.abs().idxmax()
    worst = test.loc[worst_day]
    cutoff = test.index.min()
    sns.set_theme(style='whitegrid', rc={'grid.color': GRID, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': GRID, 'xtick.color': MUTED,
        'ytick.color': MUTED, 'svg.fonttype': 'none', 'svg.hashsalt': 'bikeshare', 'pdf.fonttype': 42})

    def volume(ax):
        history = daily[daily.dteday < cutoff]
        ax.plot(history.dteday, history.cnt, color=BLUE, lw=.65, alpha=.65, label='Development: daily')
        ax.plot(history.dteday, history.cnt.rolling(28).mean(), color=TEAL, lw=2, label='Development: trailing 28-day mean')
        ax.axvspan(cutoff, daily.dteday.max(), color=ORANGE, alpha=.12, label='Final test period')
        ax.set(xlim=(daily.dteday.min(), daily.dteday.max()), ylim=(0, 9250), ylabel='Trips per day')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.legend(loc='upper left', frameon=False, fontsize=8.5)
        sns.despine(ax=ax)

    def split(ax):
        for row in result['folds'].itertuples():
            ax.barh(row.fold, row.train_days, left=mdates.date2num(row.train_start), height=.56,
                    color=BLUE, label='Train' if row.fold == 1 else None)
            ax.barh(row.fold, row.validation_days, left=mdates.date2num(row.validation_start), height=.56,
                    color=TEAL, label='Validate' if row.fold == 1 else None)
        ax.barh(5, result['fit_days'], left=mdates.date2num(result['fit_start']), height=.56, color=BLUE)
        ax.barh(5, len(test), left=mdates.date2num(cutoff), height=.56, color=ORANGE, label='Final test')
        ax.set(yticks=range(1, 6), yticklabels=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Final'],
               xlim=(result['fit_start'], test.index[-1]+pd.Timedelta(days=1)))
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.yaxis.grid(False)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, fontsize=9)
        sns.despine(ax=ax)

    def compare(ax):
        table = result['average'].sort_values().rename('MAE').rename_axis('Model').reset_index()
        sns.barplot(data=table, x='MAE', y='Model', hue='Model', legend=False, ax=ax,
                    palette={name: TEAL if name == result['winner'] else BLUE for name in table.Model})
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.0f', padding=5, fontsize=10)
        ax.set(xlim=(0, table.MAE.max()*1.17), xlabel='Mean validation MAE (trips/day)', ylabel='')
        sns.despine(ax=ax)

    def by_fold(ax):
        table = result['cv'].pivot(index='model', columns='fold', values='mae').reindex(result['average'].sort_values().index)
        table.columns = [f'Fold {i}' for i in table.columns]
        sns.heatmap(table, cmap=sns.light_palette(TEAL, as_cmap=True, reverse=True), annot=True, fmt='.0f',
                    cbar=False, linewidths=1, linecolor='white', ax=ax)
        ax.set(xlabel='', ylabel='')
        ax.tick_params(axis='y', rotation=0)

    def actual_forecast(ax):
        ax.plot(test.index, test.baseline, color=BLUE, alpha=.48, lw=.85, ls='--', label=result['benchmark'])
        ax.plot(test.index, test.cnt, color=INK, lw=1, label='Observed')
        ax.plot(test.index, test.forecast, color=TEAL, lw=2, label=result['winner'])
        ax.scatter(worst_day, worst.cnt, color=ORANGE, s=27, zorder=5)
        ax.annotate(f'{worst_day:%b %d}: {worst.cnt:,.0f} observed', (worst_day, worst.cnt),
                    xytext=(14, 30), textcoords='offset points', size=9, color=ORANGE,
                    arrowprops={'arrowstyle': '-', 'color': ORANGE})
        ax.set(ylim=(0, 9750), xlim=(test.index.min(), test.index.max()), ylabel='Trips per day')
        ticks = [test.index.min(), *pd.date_range(test.index.min(), test.index.max(), freq='MS')]
        ax.set_xticks(ticks, [day.strftime('%b') for day in ticks])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.03), frameon=False, ncol=3, fontsize=9)
        sns.despine(ax=ax)

    def errors(ax):
        ax.bar(test.index, test.error, color=np.where(test.error >= 0, ORANGE, BLUE), width=.85)
        ax.axhline(0, color=INK, lw=.85)
        ax.set(xlim=(test.index.min()-pd.Timedelta(days=1), test.index.max()+pd.Timedelta(days=1)),
               ylabel='Forecast minus actual (trips)', xlabel='Date in final test / 2012')
        ticks = [test.index.min(), *pd.date_range(test.index.min(), test.index.max(), freq='MS')]
        ax.set_xticks(ticks, [day.strftime('%b') for day in ticks])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        sns.despine(ax=ax)

    charts = [('01-demand-history', (8, 3.5), volume), ('02-training-timeline', (8, 3.5), split),
              ('03-validation-ranking', (8, 3), compare), ('04-fold-errors', (8, 2.8), by_fold),
              ('05-actual-vs-forecast', (8, 3.6), actual_forecast), ('06-forecast-errors', (8, 3), errors)]
    for name, size, draw in charts:
        chart(name, size, draw)

    with PdfPages(OUT / 'bikeshare-forecast.pdf', metadata={'Title': 'Forecasting tomorrow’s bike rentals',
                  'Subject': 'Capital Bikeshare historical forecasting case study', 'CreationDate': None}) as pdf:
        fig = page(1, 'The question', 'Forecasting tomorrow’s\nbike rentals')
        note(fig, .765, 'Decision', 'Can a bike-share operator estimate tomorrow’s total trips after tonight’s counts close? '
             'A system-wide estimate could inform staffing and capacity planning, but it cannot tell crews which station needs bikes.')
        fig.text(.103, .610, f'{len(daily):,}', size=26, color=TEAL, weight='bold')
        fig.text(.374, .610, f'{len(test)}', size=26, color=TEAL, weight='bold')
        fig.text(.627, .610, f'{mean_mae:,.1f}', size=26, color=TEAL, weight='bold')
        for x, label in zip([.103, .374, .627], ['Daily records', 'Final test days', 'Test MAE / trips per day']):
            fig.text(x, .584, label, size=10, color=MUTED)
        volume(fig.add_axes([.17, .252, .72, .275]))
        note(fig, .183, 'What the history shows', 'Rental volume varies day to day and moves with the seasons. The last '
             '120 days are reserved for testing. Their rental counts are hidden in this chart to show what was available for development.')
        pdf.savefig(fig); plt.close(fig)

        fig = page(2, 'Method', 'How the models\nwere trained')
        note(fig, .752, '01 / Check the data', f'UCI Bike Sharing, day.csv [1]. One row per day, {daily.dteday.min():%b %d, %Y} through '
             f'{daily.dteday.max():%b %d, %Y}. Verified unique dates, no missing days or values, and total rentals = casual + registered. '
             'The first seven days supply history but cannot be scored.')
        note(fig, .625, '02 / Only use information available at prediction time', 'For target day t, use past rental counts '
             'and yesterday’s temperature and humidity, plus t’s known calendar. Exclude t’s recorded weather and both components of t’s rental total.')
        split(fig.add_axes([.184, .237, .699, .23]))
        note(fig, .165, '03 / Validate in date order [2]', 'Four expanding windows, each scored on the next 84 days. Refit the selected '
             'candidate on all 604 development rows if it needs fitting. During the final test, prior actual counts become available day by day.')
        pdf.savefig(fig); plt.close(fig)

        fig = page(3, 'Validation', 'A rolling average\nwon model selection')
        note(fig, .751, 'Measure error in trips, not percentages', 'Mean absolute error (MAE) is the average size of the miss, '
             'regardless of direction. Select the simplest candidate within 2% of the lowest average MAE across the four validation windows.')
        compare(fig.add_axes([.237, .438, .646, .212]))
        by_fold(fig.add_axes([.237, .192, .646, .172]))
        note(fig, .119, 'Read across the rows', 'Ridge won folds 3 and 4. The 7-day mean had the lowest average and needs no fitted parameters.')
        pdf.savefig(fig); plt.close(fig)

        fig = page(4, 'Final test', 'Yesterday’s count\nwas harder to beat')
        note(fig, .753, f'{cutoff:%b %d} to {test.index.max():%b %d, %Y}  /  {len(test)} one-day forecasts',
             f'The selected {result["winner"]} missed by {mean_mae:,.1f} trips per day on average versus {ref_mae:,.1f} for '
             f'{result["benchmark"].lower()} (+{gap:.1f}% MAE). Keep both rules as benchmarks; this test does not establish a consistent winner.')
        actual_forecast(fig.add_axes([.179, .435, .707, .20]))
        errors(fig.add_axes([.179, .204, .707, .160]))
        note(fig, .137, 'Why the average can miss',
             f'On {worst_day:%b %d}, the prior-week mean predicted {worst.forecast:,.0f} trips and only {worst.cnt:,.0f} were recorded. '
             'A backward-looking average cannot see an abrupt drop coming.')
        pdf.savefig(fig); plt.close(fig)

        fig = page(5, 'Decision', 'What to try next')
        note(fig, .752, 'Keep the simple rule as a baseline, not a deployment claim', 'The 7-day mean is cheap to compute and easy '
             'to explain, but validation and the final test disagree. Neither result tells us which rule will work next year.')
        note(fig, .623, 'Bring in information the operator would really have', 'Test archived next-day weather forecasts or known '
             'event and service notices, then score on more recent dates across multiple seasons. Do not substitute observed future weather.')
        note(fig, .494, 'Know what the target cannot tell you', 'Recorded rentals are completed usage, not unmet demand. The old dataset '
             'has no station-level target or stockout labels. The large late-October miss identifies a day to investigate, not its cause.')
        note(fig, .365, 'Reproduce and verify', 'Run python run.py to rebuild this PDF, six SVG charts and the tables. Use '
             'audit.sql to verify the test error for both rules independently in SQLite. The inputs and dependency versions are pinned in the repository.')
        fig.text(.098, .227, 'SOURCES', size=10, weight='bold', color=TEAL)
        fig.text(.098, .199, '[1] Fanaee-T (2013), Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894.',
                 size=9, color=INK, url='https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset')
        fig.text(.098, .174, '[2] scikit-learn documentation, TimeSeriesSplit.', size=9, color=INK,
                 url='https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html')
        pdf.savefig(fig); plt.close(fig)


def sql_check(daily, scores):
    table = daily[['dteday', 'cnt']].assign(dteday=daily.dteday.dt.strftime('%Y-%m-%d'))
    with sqlite3.connect(':memory:') as db:
        table.to_sql('daily', db, index=False)
        days, rolling_mae, yesterday_mae = db.execute((ROOT / 'audit.sql').read_text(encoding='utf-8')).fetchone()
    assert days == len(scores['predictions'])
    for name, expected in [('7-day mean', rolling_mae), ('Yesterday', yesterday_mae)]:
        for col, candidate in [('forecast', 'winner'), ('baseline', 'benchmark')]:
            if scores[candidate] == name:
                test = scores['predictions']
                np.testing.assert_allclose((test[col] - test.cnt).abs().mean(), expected, rtol=1e-12)
    return {'test_days': days, '7_day_mean_mae': rolling_mae, 'yesterday_mae': yesterday_mae}


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    daily = load_snapshot()
    features = make_features(daily)
    scores = evaluate(features)
    check = sql_check(daily, scores)
    scores['cv'].to_csv(OUT / 'validation_errors.csv', index=False, float_format='%.8f')
    scores['folds'].to_csv(OUT / 'fold_schedule.csv', index=False)
    test = scores['predictions']
    test.to_csv(OUT / 'test_predictions.csv', float_format='%.8f')
    summary = {'source_days': len(daily), 'modeling_days': len(features), 'development_days': scores['fit_days'],
               'test_days': len(test), 'test_start': str(test.index[0].date()), 'test_end': str(test.index[-1].date()),
               'validation_mae': scores['average'].to_dict(), 'selected': scores['winner'], 'benchmark': scores['benchmark'],
               'test_mae_selected': scores['test_mae'], 'test_mae_benchmark': scores['baseline_mae'],
               'test_signed_bias': test.error.mean(), 'largest_error_day': str(test.error.abs().idxmax().date()),
               'independent_sql_check': check}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    make_report(daily, scores)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
