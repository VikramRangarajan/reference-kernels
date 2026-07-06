import yaml
from pathlib import Path


with open("task.yml") as f:
    out = yaml.safe_load(f)

tests = out["tests"]

test_strs = []
for test in tests:
    test_str = "; ".join(f"{k}: {v}" for k, v in test.items())
    test_strs.append(test_str)

final_str = "\n".join(test_strs)
Path("tests.txt").write_text(final_str)


benchmarks = out["benchmarks"]

bench_strs = []
for bench in benchmarks:
    bench_str = "; ".join(f"{k}: {v}" for k, v in bench.items())
    bench_strs.append(bench_str)

final_str = "\n".join(bench_strs)
Path("bench.txt").write_text(final_str)