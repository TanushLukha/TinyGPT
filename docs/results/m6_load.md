# M6 — load test

arm64 Darwin, 8 CPU cores; server: 4 torch threads, device cpu. Client and server on the same machine; closed loop (each client waits for its answer before sending the next). Each row counts the requests that finish in a 10-second window, opened once every client has had its first answer. Produced by `scripts/load_test.py`.

## Parsing: batching off vs on

Each request is one Hinglish-TOP validation request. *Batching on*: the worker parses everything waiting in its queue together (up to 16) in one padded forward pass.

| clients | batching | requests/s | mean batch | p50 ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | off | 12.4 | 1.0 | 72 | 166 | 316 |
| 2 | off | 13.1 | 1.0 | 136 | 252 | 390 |
| 4 | off | 13.2 | 1.0 | 290 | 421 | 463 |
| 8 | off | 13.0 | 1.0 | 611 | 787 | 899 |
| 16 | off | 13.2 | 1.0 | 1202 | 1366 | 1421 |
| 32 | off | 12.7 | 1.0 | 2512 | 2671 | 2790 |
| 1 | on | 12.3 | 1.0 | 72 | 167 | 317 |
| 2 | on | 13.1 | 1.0 | 136 | 251 | 399 |
| 4 | on | 13.6 | 2.5 | 281 | 454 | 516 |
| 8 | on | 17.6 | 6.2 | 408 | 864 | 869 |
| 16 | on | 25.3 | 11.2 | 633 | 1046 | 1157 |
| 32 | on | 32.0 | 16.0 | 912 | 1421 | 1886 |

## Generation: one stream at a time

64-token streams (temperature 0.8, top-k 50). The worker serves one stream at a time; the others wait in its queue.

| clients | streams | tokens/s (all clients) | first token p50 ms | p95 ms | whole stream p50 ms | p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107 | 453 | 6 | 6 | 140 | 144 |
| 2 | 108 | 460 | 143 | 147 | 278 | 284 |
| 4 | 108 | 456 | 422 | 439 | 556 | 579 |
| 8 | 108 | 456 | 980 | 1007 | 1115 | 1145 |

## Overload

40 streams sent at the same moment to a server whose generation queue holds 8: 9 served (the last finished after 1.3 s), 31 refused with HTTP 503 in 22 ms (median).

Server start to first healthy response (Python imports + loading both models): 1.7 s.
