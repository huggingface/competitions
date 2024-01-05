import pandas as pd


sub = []
for i in range(10000):
    sub.append((i, 0.5))

sub = pd.DataFrame(sub, columns=["id", "pred"])
sub.to_csv("submission.csv", index=False)
