# Experiments

Experiments are organized by paper role.

- `motivation/`: real edge-cloud serving benchmark used to motivate the
  system problem and show the trade-off between target-call reduction and
  end-to-end serving overhead.
- `real_matrix/`: config-driven paper experiment runner, result processor,
  figure generator, and smoke test.
- `scheduler/`: scheduler engine and policy implementation called by
  `real_matrix`.

Each experiment folder should contain its own `README.md`, runnable scripts,
result CSVs, and figure-generation code.
