#!/usr/bin/env python3
"""
Test script to demonstrate the progress monitoring functionality
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scripts.simulation.train_prism import TrainingProgressMonitor

def test_progress_monitor():
    """Test the progress monitor with simulated training"""
    print("🧪 Testing Progress Monitor...")
    print("="*60)
    
    # Create progress monitor
    monitor = TrainingProgressMonitor(total_epochs=3, total_batches_per_epoch=50)
    
    # Simulate training
    for epoch in range(1, 4):
        monitor.start_epoch(epoch)
        
        for batch in range(50):
            monitor.start_batch(batch)
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Simulate loss
            loss = 0.5 + 0.1 * (batch / 50) + 0.05 * (epoch - 1)
            
            # Update progress
            monitor.update_batch_progress(batch, loss, 50)
        
        # End epoch
        monitor.end_epoch(loss)
    
    # Show final summary
    summary = monitor.get_performance_summary()
    print(f"\n📊 Final Summary: {summary}")

if __name__ == "__main__":
    test_progress_monitor()
