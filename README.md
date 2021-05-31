upeval
=====================
Evaluation tools for Uplift Modelling, with continuous outcome support.
Serves as a supplement to [pylift](https://github.com/wayfair/pylift).

Usage
---------

```python
import pandas as pd
from upeval import *
df = pd.DataFrame({
	"Treatment": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
	"Outcome":   [6, 7, 1, 1, 7, 8, 2, 2, 9, 10],
	"Uscore":    [0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5]
})
upe = UpContEval(df)
upq = UpContEvalQ(df, nquantiles = 3)
print(upq.auuc_score())
upq.plot_cumsums("cumsums.png")
upq.plot_quantiles("quantiles.png", density = True)
```

COPYING
---------
Copyright (c) 2021 jnjcc, [Yste.org](http://www.yste.org)

This program is licensed under the terms of the MIT license. See [COPYING](COPYING)
for more details.
