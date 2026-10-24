# Finding real disruptions in Swiss train punctuality data

Visual Analytics (van), FHNW, HS26.

Swiss public transport publishes the scheduled and the actual time for every stop of every trip,
about 1.6 million stop events a day. Somewhere in there are the days a line genuinely fell apart.
Nobody can find those by scrolling, so this is a tool where an algorithm proposes candidate
disruptions and a person decides which are real, with the evidence shown next to each proposal.

## Data

**Ist-Daten** from [opentransportdata.swiss](https://opentransportdata.swiss/en/cookbook/actual-data/),
monthly archives at
[archive.opentransportdata.swiss](https://archive.opentransportdata.swiss/actual_data_archive.htm).
Station coordinates from
[Service Points](https://data.opentransportdata.swiss/dataset/service-point-v2), joined on `BPUIC`.

Scope is trains only (`PRODUKT_ID = 'Zug'`), September 2025 to August 2026. The v2 schema starts
in July 2025, so this window is one consistent format. That is 64.4 million train stops, which
aggregate to 20.9 million cells of (line, station, hour).

A monthly archive is 1.3 GB zipped and about 17 GB unpacked, so `prepare.py` streams one day at a
time and deletes each extracted copy straight away.

## Why an algorithm is needed

Volume is the obvious reason. The less obvious one matters more: late is not the same everywhere.
The 07:40 into Zürich HB is routinely two minutes down and nobody cares, while a rural line that
normally runs exact being four minutes down is a real event. A fixed threshold finds the busy
stations every morning and misses the incidents. What matters is late compared to what that line
normally does at that hour of that weekday, and computing that for every line, station and hour
is the part the algorithm does.

## The model

Delay carries along a route, and how much each leg carries is fitted from the data:

    delay_out = a * delay_in + b

Held out on February after fitting on September to January, the per leg model predicts the
next stop to within 22.2 seconds, against 43.4 for assuming the delay is unchanged and 41.9
for one global coefficient. `a` averages 0.909 and tracks the slack in the timetable: a leg booked two minutes keeps 0.967
of whatever arrives, one booked ten keeps 0.755. 30% of legs shed delay, 67% hold it, 3%
amplify it. Fitting per leg halves the one-step error against assuming the delay is unchanged,
43.2s to 20.6s over 11.3M legs.

Applied along a route the coefficients compound, so a starting delay becomes a prediction of how
far it reaches, and the variance compounds with it. The interval widens at every stop, which is
the honest picture: five stops out is worth less than one stop out.

The band is calibrated against measured residuals rather than the normal distribution,
because the residuals are nowhere near normal: a true 95% needs about ten sigma, 21% of
stops land beyond three sigma where a normal says 0.3%, and the worst is 52. So the control
asks how often the band should hold rather than for a sigma count.

`prepare.py` stores sufficient statistics per leg rather than coefficients. A least squares fit
needs only n and five sums, and those are additive, so pooling to line level or to one global
coefficient is adding rows up. That is why every model parameter refits the model in under a
tenth of a second instead of filtering its output.

## Finding the episodes worth asking about

1. **Aggregate** to (line, station, hour), keeping three measurements that turn out to be almost
   uncorrelated: the delay **level** on arrival, the **gain** since the previous stop, and the
   **dwell** overrun at the platform.
2. **Baseline** per (line, station, hour-of-week): median and interquartile range over the
   preceding weeks. A trailing window follows timetable changes on its own.
3. **Score**: `(observed - baseline) / spread`, with a floor on the spread, plus a
   user-weighted cancellation term.
4. **Group** cells that touch in time and along the route into episodes, since an episode is what
   an analyst actually reports.
5. **Rank** by severity: the delay actually carried by the trains that ran, plus the waiting
   caused by the ones that did not. Cancellations are charged at the headway of that hour, so
   losing the only train on an hourly line costs an hour of waiting and losing one of eight
   costs seven minutes. It comes out in train-hours, and it considers magnitude, duration,
   extent and service volume without any weights to justify.

Level propagates downstream unchanged, so it cannot say where a delay began. Gain can, and it is
what names the origin of each episode.

Robust statistics rather than a learned anomaly detector, deliberately. The baseline can be drawn
on screen, every parameter means something in plain language, and the whole method is one
sentence under questioning. The module grades transparency of the algorithm, so a black box would
cost marks rather than earn them.

## What is on screen

Controls down the left, everything else in one column:

1. **Filters** across the top: scenario, train category, canton.
2. **The ranked list** of line-days the scoring flagged. Picking a row drives everything below.
3. **The forecast**, large: what the model expects along the route from where the delay was
   added, its calibrated band, what actually happened, and what that route normally runs at.
4. **Three panels** side by side. The whole day hour by hour, including the cancellation rate,
   which is the one thing delay alone cannot show. The route against the hour, coloured by how
   late or by where delay was added. Every stop on the map, coloured by its worst hour.
5. **The verdict**, last, after the evidence rather than between two halves of it.

Chart widths are fixed rather than fluid. A Bokeh figure set to stretch takes the width of the
page instead of the column it sits in, so every chart overflowed by the width of the controls.
Pinning them was the only thing that held, and it makes the report screenshots reproducible.

## What the user controls

Filters narrow which data is in play: scenario (winter or summer), train category, canton.
The rest change how the data is treated, grouped by the question each group answers.

**Which days to look at**: how far from normal, how many minutes late on top of that, how many
trains were measured and what share of them, how much cancellations count and what a
cancellation costs in waiting, how many weeks of history define normal, and what to rank by.

**What the model expects**: whether to fit each stop pair, each line or the country, how many
trains a fit needs before it stops borrowing, whether peak and off peak are fitted apart, how
often the band should hold, and where the forecast starts.

Every control carries a plain-language label and a one-line explanation on hover, because a
parameter nobody can name is not a parameter the user steers.

The person then judges whether the model got it right and writes a note. The judgement is
appended to `data/incident_log.jsonl` together with what the model predicted, what actually
happened, and the settings in force at the time, because a verdict means nothing without the
model that produced it. Episodes already judged are marked in the table. That record is what
makes this a loop rather than a report generator.

## Two things in the data worth knowing

**Not every "actual time" is measured.** `AN_PROGNOSE_STATUS` distinguishes `REAL` from
`GESCHAETZT` and `PROGNOSE`, and an empty field means forecast. Only 77.9% of arrivals are
measured. Every number therefore carries two quality dimensions that are already in the data: how
many trips are behind it, and how many were actually measured. Both are shown, and cells below
the thresholds are drawn faded rather than hidden.

**Delay can only be measured on trains that ran.** A cancelled trip contributes no delay, so when
a line is suspended the metric does not spike, it goes quiet and reads normal on a handful of
observations. On R43 on 16 February the morning ran with half the stops cancelled and 10 to 18
minute delays, while the afternoon had 86% cancelled and the three trains that ran were on time.
Looking only at delay, the afternoon was the better half of the day.

## Environment

```bash
conda env create -f environment.yml
conda activate ml-bachelor-van
```

Python 3.11, which is what Datashader and numba are happiest on. The base conda install is 3.14,
where numba has no wheels yet.

## Running it

```bash
python src/fetch.py --start 2025-09 --end 2026-08   # about 15 GB
python src/prepare.py                                # writes data/processed/
python notebooks/01_explore.py --month 2026-02       # the report figures
panel serve src/app.py --show                        # the interface
pytest                                               # checks the two scoring paths agree
```

`fetch.py` skips what it already has, so it is safe to interrupt. `prepare.py` prints the
measured-versus-estimated breakdown, the cancellation share and the station join coverage, so
silent data loss is visible. Use `--months 2026-02` to work on one month first.

Raw and processed data are not committed. Both scripts rebuild them.

## Structure

```
src/fetch.py       download the monthly archives
src/prepare.py     archives to parquet, then the hourly cube, baselines, stop order
                   and the per-leg statistics the propagation model is fitted from
src/model.py       baseline, scoring, episode grouping. no interface code
src/propagation.py the fitted delay propagation model. no interface code
src/app.py         the Panel interface
src/style.py       shared figure styling
notebooks/         exploration, a check that the ranking works, the layout sketch,
                   04_method.ipynb on the statistics, 05_model_quality.ipynb on what
                   the model can and cannot do
tests/             the SQL and pandas scoring paths must agree
evaluation/        protocol, consent, questionnaire. Answers and analysis once it is run
figures/           before/ and after/ the user study
```

`model.py` imports nothing from Panel, so the notebooks, the tests and the app call the same
functions and the report figures regenerate without starting a server.

## Performance

The app scores in DuckDB and brings only the cells that survive the gates into pandas. Scoring
all 5.2 million cells of a scenario in pandas to keep 19,000 took 3 to 4 seconds per parameter
change; doing it this way takes 0.2 to 0.9 seconds. The baselines are precomputed for three
window lengths, 47 seconds for the whole year, which is why the window is a choice of three
rather than a free slider.

The formula exists in both SQL and pandas, so `tests/` checks that they flag the same cells.

## Figures

`figures/before/` holds the current set: the delay distribution by source of the recorded time,
the daily cancellation question, one incident hour by hour, the same incident along the route,
and the planned layout.

## AI use

| Tool | What for | Where |
| --- | --- | --- |
| Claude | Weighing dataset options against the module requirements | Choice of dataset |
| Claude | Writing the pipeline and the scoring functions | `src/` |
| Claude | Checking whether claims in my text hold up | Report |

I write the report text myself.
