# UE Antenna Configuration Guide

This guide explains how to configure the number of UE antennas to use in training and testing.

## Overview

The system now supports dynamic configuration of UE antenna usage through the `ue_antenna_count` parameter. This allows you to:

1. **Single UE Antenna Mode** (`ue_antenna_count: 1`): Use only one specific UE antenna (original behavior)
2. **Multi-UE Antenna Mode** (`ue_antenna_count > 1`): Use multiple UE antennas as additional subcarriers

## Configuration Parameters

### Data Configuration

```yaml
input:
  ue_antenna_index: 0        # Which UE antenna to use when ue_antenna_count=1
  ue_antenna_count: 1        # Number of UE antennas to use
```

### Parameters Explained

- **`ue_antenna_index`**: When `ue_antenna_count=1`, specifies which UE antenna to use (0-indexed)
- **`ue_antenna_count`**: 
  - `1`: Use single UE antenna (original behavior)
  - `>1`: Use multiple UE antennas as subcarriers

## Usage Examples

### Example 1: Single UE Antenna (Default)

```yaml
input:
  ue_antenna_index: 0
  ue_antenna_count: 1
```

**Behavior**: Uses only UE antenna 0, data shape: `[batch, bs_antennas, subcarriers]`

### Example 2: Multiple UE Antennas as Subcarriers

```yaml
input:
  ue_antenna_index: 0
  ue_antenna_count: 4
```

**Behavior**: Uses first 4 UE antennas, data shape: `[batch, bs_antennas, 4 * subcarriers]`

For PolyU dataset:
- Original: 64 subcarriers
- With 4 UE antennas: 4 × 64 = 256 subcarriers

## Dataset Compatibility

### PolyU Dataset
- **Available UE antennas**: 8
- **Recommended configurations**:
  - `ue_antenna_count: 1` (single antenna)
  - `ue_antenna_count: 4` (4 antennas as subcarriers)
  - `ue_antenna_count: 8` (all antennas as subcarriers)

### Chrissy Dataset
- **Available UE antennas**: 1
- **Recommended configuration**: `ue_antenna_count: 1` (only option)

### Sionna Dataset
- **Available UE antennas**: 1
- **Recommended configuration**: `ue_antenna_count: 1` (only option)

## Implementation Details

### Data Processing

When `ue_antenna_count > 1`:

1. **Data Selection**: Selects first N UE antennas from the dataset
2. **Dimension Reshaping**: Merges UE antenna dimension with subcarrier dimension
3. **Network Update**: Dynamically updates network configuration for new subcarrier count

### Memory Considerations

Using multiple UE antennas increases memory usage:
- **Single antenna**: `[batch, bs_antennas, subcarriers]`
- **N antennas**: `[batch, bs_antennas, N * subcarriers]`

### Performance Impact

- **Training time**: May increase due to larger subcarrier count
- **Memory usage**: Increases proportionally with UE antenna count
- **Model capacity**: Network automatically adapts to new subcarrier count

## Configuration Files

### Example: PolyU Multi-UE Configuration

See `configs/polyu_multi_ue.yml` for a complete example using 4 UE antennas.

### Key Changes in Multi-UE Configuration

```yaml
input:
  ue_antenna_count: 4  # Use 4 UE antennas

neural_networks:
  prism_network:
    num_subcarriers: 64  # Base count (will be updated to 256)
  radiance_network:
    num_subcarriers: 64  # Base count (will be updated to 256)
```

## Training and Testing

### Training

```bash
python scripts/train.py --config configs/polyu_multi_ue.yml
```

### Testing

```bash
python scripts/test.py --config configs/polyu_multi_ue.yml
```

## Troubleshooting

### Common Issues

1. **ValueError: Requested X UE antennas but only Y available**
   - **Solution**: Reduce `ue_antenna_count` to available number

2. **Memory errors with large UE antenna counts**
   - **Solution**: Reduce batch size or use fewer UE antennas

3. **Network configuration mismatch**
   - **Solution**: Ensure `num_subcarriers` in config matches expected count

### Validation

The system automatically validates:
- UE antenna count doesn't exceed available antennas
- Network configuration is updated for new subcarrier count
- Data dimensions are consistent throughout the pipeline

## Best Practices

1. **Start small**: Begin with `ue_antenna_count: 1` to ensure basic functionality
2. **Gradual increase**: Incrementally increase UE antenna count to find optimal balance
3. **Monitor memory**: Watch GPU memory usage with larger configurations
4. **Validate results**: Compare results between single and multi-UE configurations

## Technical Notes

- The system uses the "UE antennas as subcarriers" approach for maximum compatibility
- Network architectures remain unchanged, only subcarrier count is dynamic
- All existing loss functions and training procedures work without modification
- Spatial spectrum loss calculation respects the `orientation` configuration regardless of UE antenna count
