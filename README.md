# nowcoder
Solution for problems and exercises in nowcoder.

## Tips

### Read input:
- normal case
```python
inputs = [e for e in input().strip().split()]
```

- if inputs are list of integers
```python
integer_inputs = list(map(int, input().strip().split()))
```

- if we don't know the number of input lines
```python
import sys

inputs = []
for line in sys.stdin:
  inputs.append(line.strip().split())
```
