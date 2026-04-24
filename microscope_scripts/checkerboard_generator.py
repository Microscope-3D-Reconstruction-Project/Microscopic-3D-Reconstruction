import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages


def generate_checkerboard_pdf(square_size_mm, rows=6, cols=9):
    """
    Generate a checkerboard pattern PDF for camera calibration.

    Args:
        square_size_mm: Size of each checkerboard square in millimeters
        rows: Number of internal corners vertically (default 6 for 7x10 board)
        cols: Number of internal corners horizontally (default 9 for 7x10 board)
        output_filename: Output PDF filename
    """
    # US Letter paper dimensions in mm (8.5 x 11 in)
    PAPER_WIDTH_MM = 215.9
    PAPER_HEIGHT_MM = 279.4

    # Calculate checkerboard dimensions
    # We need rows+1 and cols+1 squares for the specified internal corners
    board_width_mm = (cols + 1) * square_size_mm
    board_height_mm = (rows + 1) * square_size_mm

    # Check if checkerboard fits on Letter paper
    if board_width_mm > PAPER_WIDTH_MM or board_height_mm > PAPER_HEIGHT_MM:
        print(
            f"Warning: Checkerboard ({board_width_mm}x{board_height_mm} mm) is larger than Letter paper ({PAPER_WIDTH_MM}x{PAPER_HEIGHT_MM} mm)"
        )
        print("Consider using a smaller square size.")
        return False

    # Convert mm to inches — figure size in inches maps 1:1 to PDF points (1 in = 72 pts).
    MM_TO_INCH = 1 / 25.4
    PAPER_WIDTH_INCH = PAPER_WIDTH_MM * MM_TO_INCH
    PAPER_HEIGHT_INCH = PAPER_HEIGHT_MM * MM_TO_INCH

    # Create figure with exact Letter dimensions. No axes — draw directly in figure
    # coordinates so physical size is tied only to the figure's inch-size.
    fig = plt.figure(figsize=(PAPER_WIDTH_INCH, PAPER_HEIGHT_INCH))

    # Center the checkerboard on the page
    offset_x_mm = (PAPER_WIDTH_MM - board_width_mm) / 2
    offset_y_mm = (PAPER_HEIGHT_MM - board_height_mm) / 2

    # Draw checkerboard pattern in figure-fraction coordinates
    for row in range(rows + 1):
        for col in range(cols + 1):
            color = "black" if (row + col) % 2 == 0 else "white"

            x_mm = offset_x_mm + col * square_size_mm
            y_mm = offset_y_mm + (rows - row) * square_size_mm  # Flip y-axis

            x_frac = x_mm / PAPER_WIDTH_MM
            y_frac = y_mm / PAPER_HEIGHT_MM
            w_frac = square_size_mm / PAPER_WIDTH_MM
            h_frac = square_size_mm / PAPER_HEIGHT_MM

            rect = patches.Rectangle(
                (x_frac, y_frac),
                w_frac,
                h_frac,
                linewidth=0,
                edgecolor="none",
                facecolor=color,
                transform=fig.transFigure,
            )
            fig.add_artist(rect)

    checkerboards_dir = os.path.join(
        os.path.dirname(__file__), "calibration_data", "checkerboards"
    )
    os.makedirs(checkerboards_dir, exist_ok=True)
    output_filename = f"checkerboard_{int(square_size_mm)}mm_{rows}rows_{cols}cols.pdf"
    output_path = os.path.join(checkerboards_dir, output_filename)

    # Save as PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)

    plt.close()

    print(f"Checkerboard PDF generated successfully!")
    print(f"  Output file: {output_path}")
    print(f"  Pattern: {rows}x{cols} internal corners ({rows+1}x{cols+1} squares)")
    print(f"  Square size: {square_size_mm} mm")
    print(f"  Board dimensions: {board_width_mm} x {board_height_mm} mm")
    print(f"\nPrint settings:")
    print(f"  - Print on US Letter paper (8.5 x 11 in)")
    print(f"  - Scale: 100% (NO SCALING / Actual Size)")
    print(f"  - Page orientation: Portrait")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate a checkerboard pattern PDF for camera calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python checkerboard_generator.py --size 25
  python checkerboard_generator.py --size 20 --rows 8 --cols 11
  python checkerboard_generator.py --size 30 --output my_checkerboard.pdf
        """,
    )
    parser.add_argument(
        "--size",
        type=float,
        required=True,
        help="Size of each checkerboard square in millimeters (e.g., 25)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of internal corner rows (default: 6, creates 7x10 board)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of internal corner columns (default: 9, creates 7x10 board)",
    )
    args = parser.parse_args()

    generate_checkerboard_pdf(args.size, args.rows, args.cols)


if __name__ == "__main__":
    main()
