# Flag Label Registry

The five evidence-state labels used everywhere in the pipeline (configs,
code, sample data, tests):

| Canonical label                  | Used for                                                                                        |
|----------------------------------|-------------------------------------------------------------------------------------------------|
| `Strong Evidence-Supports`       | external evidence directly supports the claim                                                   |
| `Strong Evidence-Refutes`        | external evidence directly conflicts with / weakens the claim                                   |
| `Weak Evidence-Metadata Only`    | only metadata-level relevance (title/abstract/venue/year)                                       |
| `No Evidence`                    | no relevant external evidence retrieved after two query-refinement rounds                       |
| `Non-verifiable Item`            | the fragment is subjective / about writing quality / not verifiable externally                  |

Note: the separator between "Evidence" and the suffix is the ASCII
hyphen-minus `-` (U+002D), matching the paper PDF, the README, and the
distillation-data example in the appendix.
