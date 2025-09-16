import os

import fitz  # PyMuPDF
from PIL import Image


def combine_pdfs_horizontally(pdf_paths, output_path):
    """
    Combines single-page PDFs horizontally.

    Args:
        pdf_paths (list): A list of paths to the PDF files.
        output_path (str): The path to save the combined PDF.
    """
    images = []
    total_width = 0
    max_height = 0

    # Render each PDF page as an image
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: File not found at {pdf_path}")
            continue
        try:
            doc = fitz.open(pdf_path)
            # Assuming single-page PDFs
            if doc.page_count > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=300)  # Higher DPI for better quality
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                images.append(img)
                total_width += img.width
                if img.height > max_height:
                    max_height = img.height
            doc.close()
        except Exception as e:
            print(f"Could not process {pdf_path}: {e}")


    if not images:
        print("No valid PDFs found to combine.")
        return

    # Create a new image with the combined width and max height
    # Assuming all images should be the same height, we can resize them
    # For now, let's just use max_height and align at top
    combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    # Paste the images side by side
    current_width = 0
    for img in images:
        # If you want to resize all to max_height, you would do it here.
        # For now, just paste them.
        combined_image.paste(img, (current_width, 0))
        current_width += img.width

    # Save the combined image as a PDF
    combined_image.save(output_path, "PDF", resolution=100.0)
    print(f"Combined PDF saved to {output_path}")


def combine_pdfs_vertically(pdf_paths, output_path):
    """
    Combines single-page PDFs vertically.

    Args:
        pdf_paths (list): A list of paths to the PDF files.
        output_path (str): The path to save the combined PDF.
    """
    images = []
    total_height = 0
    max_width = 0

    # Render each PDF page as an image
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: File not found at {pdf_path}")
            continue
        try:
            doc = fitz.open(pdf_path)
            # Assuming single-page PDFs
            if doc.page_count > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=300)  # Higher DPI for better quality
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                images.append(img)
                total_height += img.height
                if img.width > max_width:
                    max_width = img.width
            doc.close()
        except Exception as e:
            print(f"Could not process {pdf_path}: {e}")

    if not images:
        print("No valid PDFs found to combine.")
        return

    # Create a new image with the combined height and max width
    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    # Paste the images top to bottom
    current_height = 0
    for img in images:
        combined_image.paste(img, (0, current_height))
        current_height += img.height

    # Save the combined image as a PDF
    combined_image.save(output_path, "PDF", resolution=100.0)
    print(f"Combined PDF saved to {output_path}")


if __name__ == "__main__":
    # Combine AUC_FOV plots
    auc_fov_files = [
        "experiments/foragax-old-exploration/plots/auc-fov/auc_fov.pdf",
        "experiments/foragax/plots/auc-fov/auc_fov.pdf",
        "experiments/foragax-alt-exploration/plots/auc-fov/auc_fov.pdf",
    ]
    output_auc_fov_pdf = "combined_auc_fov.pdf"
    combine_pdfs_horizontally(auc_fov_files, output_auc_fov_pdf)

    # Combine Learning Curve plots
    learning_curve_files = [
        "experiments/foragax-old-exploration/plots/learning-curve/ForagaxTwoBiomeSmall.pdf",
        "experiments/foragax/plots/learning-curve/ForagaxTwoBiomeSmall.pdf",
        "experiments/foragax-alt-exploration/plots/learning-curve/ForagaxTwoBiomeSmall.pdf",
        # Add other learning curve PDFs here if needed
    ]
    output_learning_curve_pdf = "combined_learning_curve.pdf"
    combine_pdfs_vertically(learning_curve_files, output_learning_curve_pdf)
