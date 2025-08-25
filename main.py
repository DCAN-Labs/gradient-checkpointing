#!/usr/bin/env python3
"""
Main entry point for gradient checkpointing demonstrations.

Run this script to see all features and capabilities of the gradient
checkpointing library.
"""

import sys
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Gradient Checkpointing: Memory-Compute Trade-offs Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo basic           # Run basic checkpointing demo
  python main.py --demo benchmark       # Run benchmarking suite
  python main.py --demo optimal         # Run optimal checkpoint selection
  python main.py --demo architecture    # Run architecture-specific demos
  python main.py --demo profiling       # Run memory profiling demo
  python main.py --demo all            # Run all demonstrations
  python main.py --example 1           # Run specific example (1-8)
  python main.py --run-benchmarks      # Run full benchmark suite
        """
    )
    
    parser.add_argument(
        '--demo',
        choices=['basic', 'benchmark', 'optimal', 'architecture', 'profiling', 'all'],
        help='Which demonstration to run'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=range(1, 9),
        help='Run specific example (1-8)'
    )
    
    parser.add_argument(
        '--run-benchmarks',
        action='store_true',
        help='Run full benchmarking suite'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for demos'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print("="*80)
    print("Gradient Checkpointing Library")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Using device: {args.device}")
    print("="*80)
    
    # Handle different modes
    if args.demo:
        run_demo(args.demo, args.device)
    elif args.example:
        run_example(args.example)
    elif args.run_benchmarks:
        run_full_benchmarks()
    else:
        # Default: show menu
        show_interactive_menu()


def run_demo(demo_type, device):
    """Run specific demonstration."""
    
    if demo_type == 'basic':
        print("\nRunning Basic Checkpointing Demo...")
        from gradient_checkpointing import checkpoint
        from examples import example_1_basic_checkpointing
        example_1_basic_checkpointing()
        
    elif demo_type == 'benchmark':
        print("\nRunning Benchmark Demo...")
        from benchmark import compare_strategies, print_comparison_table
        results = compare_strategies(
            num_layers=8,
            hidden_size=512,
            batch_size=4,
            sequence_length=256,
            iterations=5
        )
        print_comparison_table(results)
        
    elif demo_type == 'optimal':
        print("\nRunning Optimal Checkpoint Selection Demo...")
        from optimal_checkpointing import demonstrate_optimal_checkpointing
        demonstrate_optimal_checkpointing()
        
    elif demo_type == 'architecture':
        print("\nRunning Architecture-Specific Demo...")
        from architecture_specific import demonstrate_architecture_specific
        demonstrate_architecture_specific()
        
    elif demo_type == 'profiling':
        print("\nRunning Memory Profiling Demo...")
        from profiling_visualization import demonstrate_profiling
        demonstrate_profiling()
        
    elif demo_type == 'all':
        print("\nRunning ALL Demonstrations...")
        from examples import run_all_examples
        run_all_examples()


def run_example(example_num):
    """Run specific example."""
    from examples import (
        example_1_basic_checkpointing,
        example_2_checkpointed_sequential,
        example_3_selective_checkpointing,
        example_4_optimal_checkpoint_selection,
        example_5_architecture_specific,
        example_6_memory_profiling,
        example_7_gradient_accumulation,
        example_8_mixed_precision
    )
    
    examples = {
        1: example_1_basic_checkpointing,
        2: example_2_checkpointed_sequential,
        3: example_3_selective_checkpointing,
        4: example_4_optimal_checkpoint_selection,
        5: example_5_architecture_specific,
        6: example_6_memory_profiling,
        7: example_7_gradient_accumulation,
        8: example_8_mixed_precision
    }
    
    print(f"\nRunning Example {example_num}...")
    examples[example_num]()


def run_full_benchmarks():
    """Run comprehensive benchmarks."""
    print("\nRunning Full Benchmark Suite...")
    print("This may take several minutes...\n")
    
    from benchmark import compare_strategies, print_comparison_table
    
    configurations = [
        ("Small Model", dict(num_layers=6, hidden_size=512, batch_size=8, sequence_length=256)),
        ("Medium Model", dict(num_layers=12, hidden_size=768, batch_size=4, sequence_length=512)),
    ]
    
    # Add large model if enough GPU memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
        configurations.append(
            ("Large Model", dict(num_layers=24, hidden_size=1024, batch_size=2, sequence_length=1024))
        )
    
    all_results = {}
    for name, config in configurations:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        
        results = compare_strategies(**config, iterations=10)
        print_comparison_table(results)
        all_results[name] = results
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        baseline = results.get('standard')
        for strategy, result in results.items():
            if strategy != 'standard' and baseline:
                mem_save = (1 - result.peak_memory_mb/baseline.peak_memory_mb) * 100
                compute_over = (result.total_time_seconds/baseline.total_time_seconds - 1) * 100
                print(f"  {result.strategy}: {mem_save:.1f}% memory saved, {compute_over:.1f}% compute overhead")


def show_interactive_menu():
    """Show interactive menu for demonstrations."""
    print("\n" + "="*60)
    print("Interactive Menu")
    print("="*60)
    print("\nAvailable demonstrations:")
    print("1. Basic Gradient Checkpointing")
    print("2. Checkpointed Sequential Container")
    print("3. Selective Layer Checkpointing")
    print("4. Optimal Checkpoint Selection")
    print("5. Architecture-Specific Strategies")
    print("6. Memory Profiling and Analysis")
    print("7. Gradient Accumulation")
    print("8. Mixed Precision + Checkpointing")
    print("9. Run Full Benchmark Suite")
    print("10. Run All Examples")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-10): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                run_example(int(choice))
            elif choice == '9':
                run_full_benchmarks()
            elif choice == '10':
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


if __name__ == "__main__":
    main()