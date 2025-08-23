#!/usr/bin/env python3
"""
Verification script to ensure all use_multiprocessing references are removed.
"""

import os
import subprocess

def check_file_for_multiprocessing(file_path):
    """Check a single file for use_multiprocessing references."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'use_multiprocessing' in content:
                lines = content.split('\n')
                matches = []
                for i, line in enumerate(lines, 1):
                    if 'use_multiprocessing' in line:
                        matches.append(f"  Line {i}: {line.strip()}")
                return matches
    except Exception as e:
        return [f"Error reading file: {e}"]
    return []

def main():
    print("🔍 Verifying use_multiprocessing cleanup")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        'scripts/simulation/train_prism.py',
        'src/prism/ray_tracer_cpu.py',
        'src/prism/ray_tracer_cuda.py',
        'configs/ofdm-5g-sionna.yml',
        'configs/README.md'
    ]
    
    total_issues = 0
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n📁 Checking {file_path}...")
            matches = check_file_for_multiprocessing(file_path)
            if matches:
                print(f"❌ Found {len(matches)} use_multiprocessing references:")
                for match in matches:
                    print(match)
                total_issues += len(matches)
            else:
                print("✅ Clean - no use_multiprocessing references found")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Also do a global search
    print(f"\n🌐 Global search for use_multiprocessing...")
    try:
        result = subprocess.run(
            ['grep', '-r', 'use_multiprocessing', '.', '--include=*.py', '--include=*.yml'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print("❌ Found remaining references:")
            print(result.stdout)
            # Count lines
            global_issues = len(result.stdout.strip().split('\n'))
            total_issues += global_issues
        else:
            print("✅ No use_multiprocessing references found globally")
    except Exception as e:
        print(f"⚠️  Global search failed: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    if total_issues == 0:
        print("🎉 SUCCESS: All use_multiprocessing references have been removed!")
        print("✅ Cleanup is complete")
        return True
    else:
        print(f"❌ ISSUES FOUND: {total_issues} use_multiprocessing references still exist")
        print("⚠️  Manual cleanup required")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
