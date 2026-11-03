# Report outline

Working skeleton, not the report. Each section says what it has to land, how much room it gets,
and which evidence it rests on. I write the prose into this.

Five pages of main text is the ceiling. That is roughly 1800 words once four figures are in, so
every section below has a word budget. If one runs over, something else loses the room, and the
decision about which should be made here rather than at midnight on 17 December.

Marking: design and theory 35%, evaluation and iteration 30%, documentation and reproducibility
20%, pitch 15%. The pitch is not part of this document. The 20% block is mostly graded on the
repository, so it gets the least room here.

## Page budget

| § | Section | Words | Figure |
| --- | --- | --- | --- |
| 1 | The task and who has it | 200 | |
| 2 | The data and what is wrong with it | 200 | Fig 1 |
| 3 | The algorithm | 450 | Fig 2 |
| 4 | The interface and the loop | 300 | Fig 3 |
| 5 | Evaluation | 550 | Fig 4 |
| 6 | What the study changed | 200 | before/after |
| 7 | Limits and reproducibility | 150 | |

Sections 3 and 5 carry 65% of the mark between them and get 55% of the words. That is deliberate.

---

## 1. The task and who has it (200 w)

**Has to land:** why a person is needed at all, and why an algorithm is needed at all.

- Service quality analyst at an operator. Question: which line-days had a real disruption, where
  did it start, how far did it spread, is the evidence good enough to report.
- Volume is the cheap argument: 1.6M stop events a day.
- The argument that actually matters: late is not the same everywhere. The 07:40 into Zürich HB
  is routinely two minutes down and nobody cares. A rural line that normally runs exact being
  four minutes down is an event. A fixed threshold finds the busy stations every morning and
  misses the incidents.
- One sentence saying the person decides, the algorithm proposes. That frames the whole report.

**Do not** spend words introducing public transport or explaining what a delay is.

## 2. The data and what is wrong with it (200 w) with Fig 1

**Has to land:** that I know what the data does not tell me.

- Ist-Daten, opentransportdata.swiss, Sep 2025 to Aug 2026, trains only. 64.4M stops, 20.9M cells
  of (line, station, hour). One sentence on scope, no more.
- **Only 77.9% of arrivals are measured.** The rest are estimated or forecast. Every number
  therefore carries two quality dimensions that are already in the data.
- **Delay can only be measured on trains that ran.** A cancelled trip contributes no delay, so a
  suspended line goes quiet rather than spiking. Use the R43 case, 16 February: the morning ran
  with half the stops cancelled and 10 to 18 minutes of delay, the afternoon had 86% cancelled
  and the three trains that ran were on time. Looking only at delay, the afternoon was the
  better half of the day.
- **Line codes are not unique.** S9 is five unrelated railways. 81 of 249 codes touch four or
  more cantons. Say what I did about it and what is still wrong.

**Fig 1:** `figures/before/01_delay_by_status.png`, the delay distribution split by source of the
recorded time. It makes the measurement-quality point without a sentence.

The R43 case or the line codes, not both at full length. Whichever one survives the first draft.

## 3. The algorithm (450 w) with Fig 2

This is the 35%. It needs the most care and it is where a reader decides whether the work is
serious.

**3a. Finding episodes (150 w)**

- Aggregate to (line, station, hour), keeping three measurements: level on arrival, gain since
  the previous stop, dwell overrun. They turn out to be almost uncorrelated.
- Baseline per (line, station, hour-of-week), median and IQR over a trailing window.
- Score as (observed - baseline) / spread, with a floor, plus a cancellation term.
- Group adjacent cells into episodes, because an episode is what an analyst reports.
- **Level propagates downstream unchanged, so it cannot say where a delay began. Gain can.**
  That sentence earns its place: it is why there are two colourings and it is what task 2 in the
  study tests.

**3b. The model (200 w)**

- `delay_out = a * delay_in + b`, fitted per leg.
- Held out on February after fitting September to January: **22.2 s per stop against 43.4 s for
  assuming the delay is unchanged and 41.9 s for one global coefficient.** The comparison is the
  claim, not the 22.2 on its own.
- `a` averages 0.909 and tracks the slack in the timetable: a leg booked two minutes keeps 0.967
  of what arrives, one booked ten keeps 0.755. 30% of legs shed delay, 67% hold it, 3% amplify.
- Sufficient statistics: a least squares fit needs n and five sums, and sums are additive, so
  pooling to line or global level is adding rows up. **That is why every model parameter refits
  the model in under a tenth of a second instead of filtering its output.** This is the one
  implementation detail worth a sentence, because it is what makes the parameters real.

**3c. Why robust statistics and not a learned detector (100 w)**

- The baseline can be drawn on screen. Every parameter means something in plain language. The
  method is one sentence under questioning.
- State the trade honestly: a learned detector would probably find things this misses.
- Tie it to the marking criterion on transparency rather than claiming it is better.

**Fig 2:** the forecast panel from the app, one route, band and actual. Caption carries the 22.2
against 43.4. If space is tight this doubles as Fig 3.

## 4. The interface and the loop (300 w) with Fig 3

**Has to land:** that the parameters are real and that uncertainty is visible.

- Four bands: filters, the ranked list, the forecast, then the route heatmap, the day hour by
  hour and the map. One sentence each at most.
- **What the user steers**, and that each control refits rather than filters.
- **Uncertainty on screen**, the part the study probes: the band widens at every stop, thin cells
  are faded rather than hidden, hours with nothing measured are grey rather than white, the
  tooltip carries the measured count.
- **The band is calibrated against 1136 measured stops, not against the normal distribution.**
  A true 95% needs about ten sigma here, not two. 21% of stops land beyond three sigma where a
  normal says 0.3%, worst case 52. So the control asks how often the band should hold rather than
  for a sigma count. This is the strongest single paragraph available and it belongs here.
- The verdict is appended with the settings in force at the time, because a verdict means nothing
  without the model that produced it.

**Fig 3:** `figures/before/05_app_layout.png`, or a screenshot of the running app, which is
better. Annotate it. An unannotated screenshot is wasted space.

## 5. Evaluation (550 w) with Fig 4

This is the 30%. Written after the study, but the design half can be written now and must be,
because writing it afterwards invites choosing the tasks that flattered the tool.

**5a. Design (200 w, writable now)**

- Within subjects, 8 participants, two conditions, counterbalanced. With and without the forecast
  panel, everything else identical.
- Three tasks on different line-days each time. Say the cases.
- **Task 3 is the one the study exists for.** R55, where the model is confidently wrong: not one
  of 13 stops inside the band, final prediction out by 17.2 minutes, and the band is a narrow
  four minutes so it looks sure of itself. The measure is not correctness but whether anyone
  raised doubt unprompted, and what they pointed at.
- Say plainly that eight people is a small sample and that the think-aloud is the stronger
  evidence.

**5b. Results (250 w, after the study)**

- Paired comparison, time and correctness on tasks 1 and 2.
- Task 3 as a count out of eight, plus what they cited.
- SUS and the three Likert items.
- **Report the result that goes against the tool as prominently as the one that favours it.**
  If both conditions score the same on tasks 1 and 2, say so and say what that means.

**5c. What the think-aloud showed (100 w)**

- Which evidence people actually mentioned. If nobody mentioned the band, that is the finding.

**Fig 4:** one chart, not three. Either the paired per-participant comparison or the task 3 count.
Decide after the data is in.

## 6. What the study changed (200 w)

The iteration half of the 30%. Needs one concrete change, made because of a finding, with the
before and after.

- Name the finding, name the change, show both states.
- One change done properly beats three mentioned.
- `figures/after/` holds the after state. Tag `v2-post-eval` makes it checkable.

## 7. Limits and reproducibility (150 w)

- The pipeline rebuilds from nothing. Two commands. Tests check that the SQL and pandas scoring
  paths agree.
- Limits, stated flat, no hedging:
  - Line codes still merge unrelated railways at the episode level. I fixed the display, not the
    grouping.
  - `min_excess` never bites at the default settings, because the spread floor times the
    threshold equals it exactly.
  - `min_obs` barely matters: 22.2 s at 10 observations, 23.4 s at 1000.
  - Eight participants, all data visualisation students, none of them railway people.

---

## What does not fit, and where it goes instead

Naming these now so they do not creep back in.

| Cut | Where it lives |
| --- | --- |
| How the pipeline is built, DuckDB, partitioning | README |
| The 3 to 4 s against 0.2 to 0.9 s rescore timing | one clause in §3b, not a paragraph |
| Summer worse than winter (June 8.58 per 1000 cells over 5 min against December 4.41) | pitch, it is a good slide and a bad paragraph |
| That forecasting tomorrow's delay is not learnable (all baselines tied near 0.19 MAE) | one sentence in §3b justifying a spatial model, or cut |
| The evidence-bucket confound | notebook 05 only |
| Every parameter explained | the app explains them, that is the point |

## Order of writing

1. §3 first. It is the longest, the most marks, and everything else refers to it.
2. §2 and §4, which are mostly already written in the README and need rewriting rather than
   inventing.
3. §5a before the study runs. Non-negotiable.
4. §1 last. An introduction written first describes a report that does not exist yet.
5. §5b, §5c and §6 after the study.
