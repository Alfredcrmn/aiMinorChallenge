# EDA findings

Generated with the same development/test split as validate_models.py. The HTML profile uses fg-data-profiling on every development window, with all 72 features and a categorical exercise label. IDs and window numbers are excluded from profiling. Test values are not profiled.

- Development: 23,168 windows, 2,896 recordings, 15 classes.
- Missing feature values: 0; infinite values: 0.
- Constant feature columns: 0.
- Duplicate feature vectors (beyond first occurrence, ignoring IDs and labels): 1043.
- Largest/smallest class ratio: 2.25.

Class counts are shown at both recording and window level; windows are correlated observations, not independent participants. Macro F1 prevents larger exercise classes from dominating the primary score.

| Exercise | Recordings | Windows |
|---|---:|---:|
| 000 | 142 | 1136 |
| 001 | 154 | 1232 |
| 002 | 136 | 1088 |
| 003 | 160 | 1280 |
| 004 | 221 | 1768 |
| 005 | 225 | 1800 |
| 006 | 169 | 1352 |
| 007 | 306 | 2448 |
| 008 | 239 | 1912 |
| 009 | 245 | 1960 |
| 010 | 199 | 1592 |
| 011 | 151 | 1208 |
| 012 | 161 | 1288 |
| 013 | 219 | 1752 |
| 015 | 169 | 1352 |

## Feature dependence

Largest absolute Pearson correlations:
- `yaw1_std` / `yaw1_range`: 0.9817.
- `roll1_std` / `roll1_range`: 0.9785.
- `yaw2_std` / `yaw2_range`: 0.9783.
- `roll2_std` / `roll2_range`: 0.9772.
- `pitch1_std` / `pitch1_range`: 0.9759.

Mean, standard deviation and RMS are mathematically related (RMS² = mean² + standard deviation²). Correlated descriptors can be redundant, so 72 columns do not imply 72 independent measurements. No feature selection was based on this report. Any future selection must be fitted within training folds.

Zero or constant windows can reflect rest or source padding, while extreme values may reflect real movement. Without valid-length masks or clinical quality labels, they are not automatically deleted or clipped. Likewise, identical feature vectors do not establish duplicate recordings.

The HTML report includes distributions, missing-value summaries, Pearson correlations, duplicate checks, and automated alerts. Pairwise scatter plots are disabled to keep the 72-feature report manageable.
