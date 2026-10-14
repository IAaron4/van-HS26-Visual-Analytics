# Evaluation protocol

What I am testing, with whom, and how. Written before running anything so the tasks are not
chosen to flatter the tool.

## The question

Two separate things, as the module asks:

1. **Usability.** Can someone use this without being taught it?
2. **Does the algorithm help?** Does the fitted propagation model make a person better at
   judging whether a disruption behaved normally, and do they notice when it should not be
   trusted?

The second matters more. A prediction nobody understands is not support.

## Design

Within subjects, 8 participants, two conditions, order counterbalanced so learning does not
favour either one.

| | what the participant sees |
| --- | --- |
| **Without the model** | the episode list, the route heatmap and the map. No forecast panel |
| **With the model** | the same, plus the forecast: expected delay along the route with its band, drawn over what actually happened |

Everything else is identical, including the filters. If the model condition also had better
filtering, any improvement could not be attributed to the model, which is the thing being
measured.

Each participant does three tasks in each condition, on **different line-days**, so nothing is
answered from memory. Tasks are matched for length of route and size of disruption.

## Participants

Eight fellow students, in person, thinking aloud, screen and audio recorded with written
consent. They are data visualisation students, not railway people, so each session opens with
two minutes of framing: what a delay is, what a line is, what the analyst is pretending to do.
No task may need more background than that.

Recruiting from the course means they read charts well and know nothing about trains, which is
the right way round: it tests the design, not domain knowledge.

## Tasks

Each task is a line-day already selected from the data, not invented. The cases were picked by
measuring how the model actually behaved on them.

### Task 1, does the disruption clear?

> *This line had a disruption. By the end of the route, had the delay recovered or not?*

Case: **S9, Schaffhausen to Uster, 26 January, from 15:00.** The model tracks reality closely
here: 95% of stops fall inside the band and the final prediction misses by half a minute.

Measures whether the model speeds up a judgement it gets right.

### Task 2, where did it start?

> *Point at the stop where this delay was generated, rather than where it was merely carried.*

Case: **S60, Ponte Tresa to Lugano, 20 January, from 09:00.**

The level view cannot answer this, since delay propagates downstream unchanged. The gain view
and the model's starting point both can. Tests whether people find that distinction.

### Task 3, the one that matters

> *The model predicts this delay will have mostly cleared by the end. Would you act on that?*

Case: **R55, La Cure to Nyon, 12 January, from 13:00.** The model is **confidently wrong**: not
one of the 13 stops falls inside the band, the final prediction misses by 17.2 minutes, and the
band is a narrow 4 minutes, so it looks sure of itself. Over half the stops rest on fewer than
three measured trains.

This is the task the study exists for. Everything on screen that is meant to convey uncertainty
is present: the widening band, the faded cells, the measured counts in the tooltip, the
fallback level on each leg. If nobody uses any of it and they simply trust the line, then the
uncertainty design has failed and that is the finding to act on.

A reserve case if R55 proves too obvious: **R71, Aigle to Les Diablerets, 6 January**, where
every stop rests on thin evidence but the prediction happens to be close, so the display says
"do not trust this" while the answer is fine.

## Why the band is worth probing

Measured across 46 episodes, the 1.96 sigma band covers **72%** of stops, where an honest
interval would cover about 95%. The model is overconfident, because the variance propagation
treats each leg's error as independent when a bad day is bad at every stop.

I would rather report that than quietly widen the band. It also means task 3 is not a contrived
trap: the tool really does present confident predictions that are wrong, and finding out whether
people notice is exactly what the evaluation is for.

## Measures

| | how |
| --- | --- |
| Time to answer | from task read out to answer given |
| Correctness | against the recorded ground truth for that line-day |
| Usability | SUS, ten items, at the end of both conditions |
| Understanding and trust | three Likert items, below |
| Reasoning | think aloud, transcribed, coded for which evidence was mentioned |

For task 3 the measure is not correctness but **whether the participant raised any doubt
unprompted**, and if so, what they pointed at.

### The three targeted items

1. I could tell how much the prediction was to be trusted.
2. I understood why the model expected what it did.
3. I would act on this prediction in a real job.

## Procedure, about 45 minutes

1. Consent, recording explained and agreed. (5 min)
2. Framing: delays, lines, the analyst's job. No mention of how the model works. (2 min)
3. First condition: a two minute look around, then three tasks thinking aloud. (15 min)
4. SUS and the three items. (3 min)
5. Second condition: same shape, different line-days. (15 min)
6. SUS and the three items again, then a short open question: when would you not trust it? (5 min)

Piloted twice before the first real session. The first pilot usually only finds wording
problems, so one is not enough.

## Analysis

Paired comparison within participants, since each does both conditions. With eight people this
is a small sample and the write up will say so: the think aloud is the stronger evidence, and
the numbers are support rather than proof.

Task 3 is reported as a count, how many of eight raised doubt unprompted, and what they cited.

## Data protection

FHNW rules apply and external AI tools are not licensed for participant data. Recordings and
transcripts are anonymised before any analysis, participants are identified as P1 to P8, and
nothing identifiable goes into the repository. Consent is written and covers the recording.

## What would make me change the design

- If both conditions score the same on tasks 1 and 2, the model is not helping and the report
  should say so rather than looking for a better metric.
- If everyone spots the bad prediction in task 3 immediately, the uncertainty display works and
  the next question is whether it works when they are hurried.
- If nobody does, the finding drives the revision for Stage 4, and that is the most useful
  outcome this study can have.
