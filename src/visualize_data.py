import numpy as np
from .constants import WAVELENGTHS
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def create_rgb(hsi_data, r_band=650, g_band=550, b_band=450):
    """
    Create an (H, W, 3) RGB image from a hyperspectral cube,
    performing per-channel min-max normalization while handling NaN and infinite values.
    """
    # pick nearest bands
    r_idx = np.argmin(np.abs(WAVELENGTHS - r_band))
    g_idx = np.argmin(np.abs(WAVELENGTHS - g_band))
    b_idx = np.argmin(np.abs(WAVELENGTHS - b_band))

    # stack into H×W×3 and cast to float32
    rgb = np.stack([hsi_data[r_idx], hsi_data[g_idx], hsi_data[b_idx]], axis=-1).astype(
        np.float32
    )

    return rgb


def show_interactive_image_with_spectrum(hsi_data, r_band=650, g_band=550, b_band=450):
    """
    Show an interactive image with a spectrum plot.
    """
    rgb = create_rgb(hsi_data, r_band, g_band, b_band)

    # Create a figure with two subplots
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the RGB image
    ax_img.imshow(rgb)
    ax_img.set_title("Click Pixels to view Spectrum")
    scatter_plot = ax_img.scatter([], [], c=[], s=100, marker="+", edgecolors=[])

    selected_pixel = []
    lines = []
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    def on_clear_button_click(event):
        # Clear the scatter plot
        selected_pixel.clear()
        scatter_plot.set_offsets(np.empty((0, 2)))
        scatter_plot.set_edgecolors([])
        for line in lines:
            line.remove()
        lines.clear()
        legend = ax_spec.legend()
        if legend:
            legend.remove()
        fig.canvas.draw()

    clear_button = widgets.Button(description="Clear Selection")
    clear_button.on_click(on_clear_button_click)
    display(clear_button)

    def on_click(event):
        if event.inaxes == ax_img:
            # Get the pixel coordinates
            x, y = int(event.xdata), int(event.ydata)

            pixel_coords = (y, x)

            # Check if the pixel is already selected
            if pixel_coords in selected_pixel:
                selected_pixel.remove(pixel_coords)
            else:
                selected_pixel.append(pixel_coords)
            update_plot()

    def update_plot():
        offsets = np.array([[px[1], px[0]] for px in selected_pixel])
        scatter_plot.set_offsets(offsets)
        marker_colors = [colors[i % len(colors)] for i in range(len(selected_pixel))]
        scatter_plot.set_edgecolors(marker_colors)
        scatter_plot.set_facecolors("none")

        for line in lines:
            line.remove()
        lines.clear()

        for idx, (py, px) in enumerate(selected_pixel):
            spectrum = hsi_data[:, py, px]
            (line,) = ax_spec.plot(
                WAVELENGTHS,
                spectrum,
                color=marker_colors[idx],
                label=f"Pixel {px},{py}",
                linestyle="-",
                marker="o",
            )
            lines.append(line)
        update_legend()
        fig.canvas.draw_idle()

    def update_legend():
        # Clear the previous legend
        legend = ax_spec.legend()
        if legend:
            legend.remove()

        # Create a new legend with the current lines
        if lines:
            ax_spec.legend(lines, [line.get_label() for line in lines], loc="upper right")
    
    ax_spec.set_title("Pixel Intensity Across Channels")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity")
    ax_spec.grid(axis="y", linestyle="--", alpha=0.7)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
