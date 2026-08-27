# Demo Examples README

This file demonstrates the proposed structure for `examples/README.md`.
The real `examples/README.md` should keep the latest validation time and the example classification matrix, then link to `status_directions.md` for detailed status definitions.

For status meanings, badge definitions, and broken-status subtypes, see [`status_directions.md`](status_directions.md).

**Last T2/T3 Validation Time:** `2026-07-07 06:55 UTC`

## Example Classification Matrix

The matrix uses one row per benchmark unit. Benchmark units belonging to the
same Example share one merged `Example` cell.

<table>
  <thead>
    <tr>
      <th>Example</th>
      <th>Benchmark Unit</th>
      <th>Path</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Demo Multi-Benchmark Example</td>
      <td>demo_runnable</td>
      <td><code>examples/demo/runnable</code></td>
      <td><img alt="Runnable" src="https://img.shields.io/badge/status-Runnable-brightgreen"></td>
    </tr>
    <tr>
      <td>demo_broken</td>
      <td><code>examples/demo/broken</code></td>
      <td><img alt="Broken" src="https://img.shields.io/badge/status-Broken-red"></td>
    </tr>
    <tr>
      <td>Demo CI State</td>
      <td>demo_cicd_ongoing</td>
      <td><code>examples/demo/cicd-ongoing</code></td>
      <td><img alt="CI/CD onGoing" src="https://img.shields.io/badge/status-CI%2FCD%20onGoing-lightgrey"></td>
    </tr>
    <tr>
      <td>Demo Example State</td>
      <td>demo_example_ongoing</td>
      <td><code>examples/demo/example-ongoing</code></td>
      <td><img alt="Example onGoing" src="https://img.shields.io/badge/status-Example%20onGoing-yellow"></td>
    </tr>
  </tbody>
</table>
