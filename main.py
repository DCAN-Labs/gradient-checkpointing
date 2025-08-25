#!/usr/bin/env python3
"""
Main entry point for gradient checkpointing demonstrations in 3D medical imaging.

Run this script to see all features and capabilities of gradient checkpointing
for medical imaging models like 3D U-Net, V-Net, and nnU-Net.
"""

import sys
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Gradient Checkpointing for 3D Medical Imaging: Memory Optimization Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo unet            # Run 3D U-Net checkpointing demo
  python main.py --demo vnet            # Run V-Net checkpointing demo  
  python main.py --demo benchmark       # Run medical imaging benchmarks
  python main.py --demo optimal         # Run optimal checkpoint selection for MRI
  python main.py --demo clinical        # Run clinical scenario demos
  python main.py --demo all            # Run all medical imaging demonstrations
  python main.py --example 1           # Run specific example (1-6)
  python main.py --run-benchmarks      # Run full medical benchmark suite
  python main.py --volume-size 128 128 128  # Specify custom volume size
        """
    )
    
    parser.add_argument(
        '--demo',
        choices=['unet', 'vnet', 'benchmark', 'optimal', 'clinical', 'all'],
        help='Which medical imaging demonstration to run'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=range(1, 7),
        help='Run specific medical imaging example (1-6)'
    )
    
    parser.add_argument(
        '--run-benchmarks',
        action='store_true',
        help='Run full medical imaging benchmarking suite'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for demos'
    )
    
    parser.add_argument(
        '--volume-size',
        nargs=3,
        type=int,
        default=[128, 128, 128],
        help='3D volume size (D H W) for testing'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for medical volume processing'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print("="*80)
    print("Gradient Checkpointing for 3D Medical Imaging")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Using device: {args.device}")
    print(f"Volume size: {tuple(args.volume_size)} voxels")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
    # Handle different modes
    if args.demo:
        run_medical_demo(args.demo, args.device, tuple(args.volume_size), args.batch_size)
    elif args.example:
        run_medical_example(args.example)
    elif args.run_benchmarks:
        run_full_brain_mri_benchmarks(tuple(args.volume_size), args.batch_size)
    else:
        # Default: show menu
        show_brain_mri_menu(args.device, tuple(args.volume_size), args.batch_size)


def run_medical_demo(demo_type, device, volume_size, batch_size):
    """Run specific medical imaging demonstration."""
    
    if demo_type == 'unet':
        print("\n🧠 Running 3D U-Net Checkpointing Demo...")
        print("Application: Brain MRI Segmentation")
        from examples import example_1_basic_3d_unet_checkpointing
        example_1_basic_3d_unet_checkpointing()
        
    elif demo_type == 'vnet':
        print("\n❤️ Running V-Net Checkpointing Demo...")
        print("Application: Cardiac MRI Analysis")
        from examples import example_2_medical_sequential_checkpointing
        example_2_medical_sequential_checkpointing()
        
    elif demo_type == 'benchmark':
        print("\n📊 Running Brain MRI Benchmark Demo...")
        from benchmark import compare_brain_mri_strategies, print_brain_mri_comparison
        results = compare_brain_mri_strategies(
            architecture="unet",
            volume_shape=volume_size,
            batch_size=batch_size,
            iterations=5
        )
        print_brain_mri_comparison(results)
        
    elif demo_type == 'optimal':
        print("\n🎯 Running Optimal Checkpoint Selection for Brain MRI...")
        from optimal_checkpointing import demonstrate_medical_checkpointing
        demonstrate_medical_checkpointing()
        
    elif demo_type == 'clinical':
        print("\n🧠 Running Brain MRI Scenario Demonstrations...")
        run_brain_mri_scenarios(device, batch_size)
        
    elif demo_type == 'all':
        print("\n🔬 Running ALL Brain MRI Demonstrations...")
        from examples import run_all_examples
        run_all_examples()


def run_medical_example(example_num):
    """Run specific medical imaging example."""
    from examples import (
        example_1_basic_3d_unet_checkpointing,
        example_2_medical_sequential_checkpointing,
        example_3_selective_layer_checkpointing,
        example_4_optimal_checkpointing_strategy,
        example_5_mixed_precision_checkpointing,
        example_6_multi_gpu_checkpointing
    )
    
    examples = {
        1: ("3D U-Net Brain MRI Segmentation", example_1_basic_3d_unet_checkpointing),
        2: ("V-Net Cardiac Imaging", example_2_medical_sequential_checkpointing),
        3: ("nnU-Net Selective Checkpointing", example_3_selective_layer_checkpointing),
        4: ("Optimal GPU Memory Management", example_4_optimal_checkpointing_strategy),
        5: ("Mixed Precision Brain Segmentation", example_5_mixed_precision_checkpointing),
        6: ("Multi-GPU Large Cohort Training", example_6_multi_gpu_checkpointing)
    }
    
    name, func = examples[example_num]
    print(f"\n🔬 Running Example {example_num}: {name}...")
    func()


def run_brain_mri_scenarios(device, batch_size):
    """Run demonstrations of brain MRI analysis scenarios."""
    print("\n" + "="*80)
    print("Brain MRI Analysis Scenarios")
    print("="*80)
    
    scenarios = [
        {
            'name': 'Brain MRI - Tumor Segmentation',
            'volume_size': (155, 240, 240),
            'description': 'Multi-modal MRI (T1, T1+Gd, T2, FLAIR)',
            'task': 'Segment tumor, edema, and necrotic tissue'
        },
        {
            'name': 'Brain MRI - Tissue Segmentation',
            'volume_size': (128, 128, 128),
            'description': 'T1-weighted structural MRI',
            'task': 'Segment gray matter, white matter, CSF'
        },
        {
            'name': 'Brain MRI - Parcellation',
            'volume_size': (256, 256, 256),
            'description': 'High-resolution structural MRI',
            'task': 'Parcellate brain into 100+ anatomical regions'
        },
        {
            'name': 'Brain MRI - Lesion Detection',
            'volume_size': (182, 218, 182),
            'description': 'T2-FLAIR MRI sequence',
            'task': 'Detect and segment white matter lesions'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Volume: {scenario['volume_size']}")
        print(f"Description: {scenario['description']}")
        print(f"Task: {scenario['task']}")
        print(f"{'='*60}")
        
        # Calculate memory requirements
        voxels = scenario['volume_size'][0] * scenario['volume_size'][1] * scenario['volume_size'][2]
        memory_gb = (voxels * 4 * 32 * batch_size) / (1024**3)  # Rough estimate
        
        print(f"Memory estimate: {memory_gb:.2f} GB")
        
        if memory_gb > 8:
            print("⚠️ Large memory requirement - Gradient checkpointing ESSENTIAL")
            print("Recommended strategy: Full encoder checkpointing for brain analysis")
        elif memory_gb > 4:
            print("⚠️ Moderate memory requirement - Checkpointing recommended")
            print("Recommended strategy: Selective checkpointing (brain region bottlenecks)")
        else:
            print("✅ Manageable memory requirement")
            print("Recommended strategy: Optional checkpointing for larger brain batches")


def run_full_brain_mri_benchmarks(volume_size, batch_size):
    """Run comprehensive brain MRI benchmarks."""
    print("\n🔬 Running Full Brain MRI Benchmark Suite...")
    print("This may take several minutes...\n")
    
    from benchmark import compare_brain_mri_strategies, print_brain_mri_comparison
    
    configurations = [
        ("Small Brain ROI (32³)", (32, 32, 32), 4),
        ("Standard Brain MRI (128³)", (128, 128, 128), 2),
        ("High-res Brain (256x256x128)", (256, 256, 128), 1),
    ]
    
    # Add large brain volume if enough GPU memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 16e9:
        configurations.append(
            ("Whole-brain High-res (256³)", (256, 256, 256), 1)
        )
    
    all_results = {}
    for name, vol_shape, batch in configurations:
        print(f"\n{'='*80}")
        print(f"Benchmarking: {name}")
        print(f"Volume: {vol_shape}, Batch: {batch}")
        print(f"{'='*80}")
        
        try:
            results = compare_brain_mri_strategies(
                architecture="unet",
                volume_shape=vol_shape,
                batch_size=batch,
                iterations=5
            )
            print_brain_mri_comparison(results)
            all_results[name] = results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ Out of memory - This is where checkpointing is essential!")
            else:
                raise e
    
    # Summary
    print("\n" + "="*80)
    print("BRAIN MRI BENCHMARK SUMMARY")
    print("="*80)
    
    for volume_name, results in all_results.items():
        print(f"\n{volume_name}:")
        baseline = results.get('standard')
        for strategy, result in results.items():
            if strategy != 'standard' and baseline:
                mem_save = (1 - result.peak_memory_mb/baseline.peak_memory_mb) * 100
                speed_ratio = result.voxels_per_second / baseline.voxels_per_second
                print(f"  {result.strategy}: {mem_save:.1f}% memory saved, {speed_ratio:.2f}x speed")


def show_brain_mri_menu(device, volume_size, batch_size):
    """Show interactive menu for brain MRI demonstrations."""
    print("\n" + "="*60)
    print("🧠 Brain MRI Analysis Interactive Menu")
    print("="*60)
    print("\nAvailable demonstrations:")
    print("1. 3D U-Net Brain MRI Segmentation")
    print("2. Brain Parcellation Sequential Checkpointing")
    print("3. Brain Tumor Detection Selective Checkpointing")
    print("4. Optimal GPU Memory Management")
    print("5. Mixed Precision Brain Segmentation")
    print("6. Multi-GPU Brain MRI Cohort Training")
    print("7. Run Brain MRI Scenarios")
    print("8. Run Full Brain MRI Benchmark Suite")
    print("9. Run All Brain MRI Examples")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice in ['1', '2', '3', '4', '5', '6']:
                run_medical_example(int(choice))
            elif choice == '7':
                run_brain_mri_scenarios(device, batch_size)
            elif choice == '8':
                run_full_brain_mri_benchmarks(volume_size, batch_size)
            elif choice == '9':
                from examples import run_all_examples
                run_all_examples()
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


def print_medical_recommendations():
    """Print recommendations for medical imaging use cases."""
    print("\n" + "="*80)
    print("📋 Gradient Checkpointing Recommendations for Medical Imaging")
    print("="*80)
    
    recommendations = [
        {
            'gpu': 'RTX 3070/3080 (8-10GB)',
            'max_volume': '128³ with batch=2',
            'strategy': 'Selective checkpointing (encoder only)',
            'use_cases': 'Brain MRI segmentation, small organ analysis'
        },
        {
            'gpu': 'V100 (16GB)',
            'max_volume': '256³ with batch=1 or 128³ with batch=4',
            'strategy': 'Selective checkpointing (bottleneck)',
            'use_cases': 'Standard clinical imaging, multi-modal fusion'
        },
        {
            'gpu': 'RTX 3090/4090 (24GB)',
            'max_volume': '256³ with batch=2',
            'strategy': 'Optional checkpointing for larger batches',
            'use_cases': 'High-resolution brain imaging, cardiac 4D flow'
        },
        {
            'gpu': 'A100 (40-80GB)',
            'max_volume': '512³ with batch=1 or 256³ with batch=4',
            'strategy': 'Checkpointing for maximum batch size',
            'use_cases': 'Whole-body imaging, large cohort studies'
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['gpu']}:")
        print(f"  Max volume: {rec['max_volume']}")
        print(f"  Strategy: {rec['strategy']}")
        print(f"  Use cases: {rec['use_cases']}")
    
    print("\n💡 Key Insights:")
    print("  • Checkpointing enables 2-4x larger volumes")
    print("  • Memory savings: 40-60% typical, up to 80% with aggressive checkpointing")
    print("  • Speed trade-off: 20-40% slower, acceptable for memory-constrained scenarios")
    print("  • Mixed precision + checkpointing: Best combination for large volumes")


if __name__ == "__main__":
    main()