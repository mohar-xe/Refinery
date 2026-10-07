# The bill

The teacher is a free model, so the dollar figure is `$0.00` and the real
quantities are printed beside it rather than replaced by an invented price
(`LLD.md` D-013). The interesting numbers are efficiency: what it costs to
produce one *verified* trajectory, and how much is wasted on rejected runs.

| Quantity | Value |
|---|---|
| API cost | **$0.00** (free tier) |
| Teacher requests | 2828 |
| Prompt tokens | 174091 |
| Completion tokens | 34944 |
| Total tokens | 209035 |
| Teacher wall time | 1458.2s |
| Mean latency / request | 0.52s |
| Provider errors | 0 (0.0000) |
| Verified trajectories | 217 |
| Tokens per verified trajectory | 963.3 |
| Requests per verified trajectory | 13.03 |
| Tokens spent on rejected runs | 5508 |
| Waste share of all tokens | 0.0263 |

## Per-run caps (enforced before each request)

- max assistant steps: 6
- max requests: 5
- max tokens: 1500
