# heartbeatpy

Heartbeat survey analysis in python, inspired by the R packages ["heaRtbeat"](https://rdrr.io/github/jpurl/heaRtbeat/f/README.md) and ["heartbeat"](https://patrickgreen93.github.io/Heartbeat/articles/Introduction.html). For more information on heartbeat analysis, check out the [2021 Wiley Award for Excellence in Survey Research](https://www.siop.org/Research-Publications/Items-of-Interest/ArtMID/19366/ArticleID/5052/SIOP-Award-Winners-Wiley-Award-for-Excellence-in-Survey-Research)

## Installation

heartbeatpy is available on ["PyPI"](https://pypi.org/project/heartbeatpy/) and can be installed using ``` pip install heartbeatpy ```


## Example usage

heartbeatpy has one main function: `calculate_votes()` which takes in a dataframe of survey data, a list of the columns to perform heartbeat transformation on, and a standard deviation threshold.

```
import pandas as pd
import numpy as np
from heartbeatpy import create_sample_data, calculate_votes

df, _ = create_sample_data(ncols=2, nrows=10)
votes, stats = calculate_votes(data=df, columns=["Q1", "Q2"], threshold=.5)
```

The function `calculate_votes()` returns two objects, the first, `votes`, contains the data provided to the function with each specified column's likert data replaced with upvotes and downvotes:

```
votes.head()

	ID	Q1	Q2
0	1	1	-1
1	2	1	-1
2	3	-1	1
3	4	-1	1
4	5	1	-1
```

The second object returned by `calcuate_votes()` is `stats`, which contains the standard deviation for each indiviual (row) in the data.

```
stats.head()

individual_sd	individual_mean
0	1.414214	3.0
1	2.121320	2.5
2	1.414214	2.0
3	0.707107	1.5
4	0.707107	1.5
```

After finding votes, the function `calculate_summary()` can be used to generate the percent of downvotes, neutral votes, and upvotes for each question. `calculate_summary()` also calculates controversy score for each question.

```
from heartbeatpy import calculate_summary

summary = calculate_summary(votes=votes, columns=cols)

summary.head()
	Question	Upvote	Downvote	Neutral	Controversy_Score
0	Q1			0.3		0.3			0.4		1.5	
1	Q2			0.4		0.4			0.2		4.0
```

Lastly, you can use the built-in `visualize_heartbeat()` function to generate an altair stacked bar chart visualization:

```
import altair as alt
from heartbeatpy import visualize_heartbeat

chart = visualize_heartbeat(summary)
display(chart)
```
