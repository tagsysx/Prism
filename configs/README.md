# Prism Configuration Guide

## Overview
This document explains the configuration parameters for the Prism neural network-based electromagnetic ray tracing system.

## Key Concepts Clarification

### UE (User Equipment) Terminology
To avoid confusion, we use specific terminology:

- **UE Device**: A single user equipment device (e.g., a smartphone)
- **UE Antennas**: Multiple antennas on a single UE device
- **UE Positions**: Different spatial locations where the same UE device is placed

### Data Structure
Our simulation data represents:
- **1 UE Device** with **4 UE Antennas** placed at **100 different positions**
- Each position has complete MIMO channel responses: `(100, 408, 4, 64)`
  - 100 positions
  - 408 subcarriers  
  - 4 UE antennas
  - 64 BS antennas

## Configuration Parameters

### Neural Networks Section
```yaml
neural_networks:
  attenuation_decoder:
    num_ue_antennas: 4    # Number of antennas per UE device (NOT number of UE devices)
  
  radiance_network:
    num_ue_antennas: 4    # Number of antennas per UE device (NOT number of UE devices)
```

### User Equipment Section
```yaml
user_equipment:
  num_ue_antennas: 4      # Number of antennas per UE device
  antenna_config: '4x64'  # 4 UE antennas × 64 BS antennas
  # Note: Number of UE positions is determined from actual training data
```

## Important Notes

1. **`num_ue_antennas`** refers to the number of antennas on a single UE device
2. **Number of UE positions** is determined from the actual training data, not from config
3. **Each training sample represents one UE position**, not multiple UE devices
4. **The model learns position-dependent channel characteristics** for a single UE device

## Data Flow
```
Training Data: (batch_size, 408, 4, 64)
├── batch_size: Number of UE positions in this batch
├── 408: Number of subcarriers
├── 4: Number of UE antennas per device
└── 64: Number of BS antennas
```

## Common Misconceptions

❌ **Wrong**: "We have 100 UE devices"
✅ **Correct**: "We have 1 UE device at 100 different positions"

❌ **Wrong**: "num_ue: 100 means 100 devices"
✅ **Correct**: "100 positions determined from training data"

❌ **Wrong**: "num_ue: 4 means 4 devices"
✅ **Correct**: "num_ue_antennas: 4 means 4 antennas per device"
