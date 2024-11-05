from PIL import Image
import os
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm


def optimize_png(input_path, output_dir, max_colors=256):
    """
    Optimize a PNG image while preserving transparency.

    Args:
        input_path (Path): Path to input image
        output_dir (Path): Base output directory
        max_colors (int): Maximum number of colors in palette
    Returns:
        tuple: (input_path, original_size, compressed_size) or (input_path, None, None) on error
    """
    try:
        # Calculate output path maintaining directory structure
        relative_path = input_path.relative_to(input_path.parent.parent)
        output_path = Path(output_dir) / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open and process image
        with Image.open(input_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Use FASTOCTREE method for RGBA images
            optimized = img.quantize(
                colors=max_colors,
                method=2,  # FASTOCTREE method
                kmeans=0   # Disable kmeans as it's not needed with FASTOCTREE
            )

            # Convert back to RGBA to preserve transparency
            optimized = optimized.convert('RGBA')

            # Save optimized image
            optimized.save(
                output_path,
                'PNG',
                optimize=True,
                compress_level=9
            )

            # Get file sizes
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)

            return input_path, original_size, compressed_size

    except Exception as e:
        print(f"\nError processing {input_path}: {str(e)}")
        return input_path, None, None


def process_directory(input_dir, output_dir, num_processes=None):
    """
    Process all PNG files in input directory using multiple processes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all PNG files
    png_files = list(input_dir.glob('**/*.png'))
    total_files = len(png_files)

    if total_files == 0:
        print("No PNG files found in the input directory.")
        return

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free

    print(
        f"\nProcessing {total_files} files using {num_processes} processes...")

    # Create partial function with fixed output_dir
    optimize_partial = partial(optimize_png, output_dir=output_dir)

    # Process files in parallel with progress bar
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(optimize_partial, png_files),
            total=total_files,
            desc="Optimizing images",
            unit="file"
        ))

    # Calculate and display statistics
    successful = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]

    if successful:
        total_original = sum(r[1] for r in successful)
        total_compressed = sum(r[2] for r in successful)
        total_reduction = (1 - total_compressed/total_original) * 100

        print("\nProcessing Complete!")
        print(f"Time taken: {time.time() - start_time:.1f} seconds")
        print(f"\nSuccessfully processed: {len(successful)} files")
        print(f"Failed: {len(failed)} files")
        print(
            f"\nTotal size reduction: {total_original/1024/1024:.1f}MB → {total_compressed/1024/1024:.1f}MB")
        print(f"Average reduction: {total_reduction:.1f}%")

        # Print details of failed files if any
        if failed:
            print("\nFailed files:")
            for f in failed:
                print(f"- {f[0]}")
    else:
        print("\nNo files were processed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize PNG images while preserving transparency')
    parser.add_argument(
        'input_dir', help='Input directory containing PNG files')
    parser.add_argument(
        'output_dir', help='Output directory for optimized files')
    parser.add_argument('-p', '--processes', type=int,
                        help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('-c', '--colors', type=int, default=256,
                        help='Maximum number of colors (default: 256)')

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.processes)


if __name__ == '__main__':
    main()
