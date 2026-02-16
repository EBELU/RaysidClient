import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import asyncio
import sys

from RaysidClient import RaysidClient, find_device_by_name


class SpectrumGUI:
    def __init__(self, address, n_channels=1800):
        self.n_channels = n_channels
        self.x = np.arange(n_channels)
        print(address)
        self.client = RaysidClient(address)
        self.client.start()

        self.latest_spectrum = np.zeros(n_channels)

        # ================= FIGURE =================
        self.fig = plt.figure(figsize=(10, 6))
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])

        # -------- Spectrum Plot --------
        self.ax_spec = self.fig.add_subplot(self.gs[0])
        self.line, = self.ax_spec.step(self.x, self.latest_spectrum, where="post")
        self.ax_spec.set_xlabel("Channel")
        self.ax_spec.set_ylabel("Counts")
        self.ax_spec.set_title("Gamma Spectrum")

        # -------- Info Panel --------
        self.ax_info = self.fig.add_subplot(self.gs[1])
        self.ax_info.axis("off")

        self.info_text = self.ax_info.text(
            0.05, 0.9,
            "Connecting...",
            fontsize=12,
            verticalalignment="top"
        )

        # -------- Reset Button --------
        button_ax = self.fig.add_axes([0.4, 0.03, 0.2, 0.07])
        self.reset_button = Button(button_ax, "Reset Spectrum")
        self.reset_button.on_clicked(self.reset_spectrum)

        # Timer for polling device
        self.timer = self.fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.refresh)
        self.timer.start()

        self.fig.canvas.mpl_connect("close_event", self.on_close)

    # =========================================================
    # POLLING LOOP (Main Thread via Timer)
    # =========================================================
    def refresh(self):
        # ----- Spectrum -----
        spectrum_pkg = self.client.LatestSpectrum
        if spectrum_pkg:
            self.latest_spectrum = np.array(spectrum_pkg.spectrum)
            self.line.set_ydata(self.latest_spectrum)
            self.ax_spec.set_ylim(
                0,
                max(1, self.latest_spectrum.max() * 1.1)
            )

        # ----- Current -----
        current = self.client.LatestRealTimeData
        status = self.client.LatestStatusData

        lines = []

        if current:
            lines.append(f"CPS: {current.CPS:.1f}")
            lines.append(f"Dose: {current.DR:.3f} µSv/h")

        if status:
            lines.append(f"Battery: {status.battery}%")
            lines.append(f"Charging: {status.charging}")
            lines.append(f"Temp: {status.temperature:.1f} °C")

        if not lines:
            lines.append("Waiting for data...")

        self.info_text.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    # =========================================================
    # RESET
    # =========================================================
    def reset_spectrum(self, event):
        try:
            self.client.reset()
        except Exception as e:
            print("Reset failed:", e)

    # =========================================================
    # CLEAN SHUTDOWN
    # =========================================================
    def on_close(self, event):
        self.timer.stop()
        self.client.stop()


# =============================================================
# ENTRY
# =============================================================
if __name__ == "__main__":
    try:
        device = asyncio.run(find_device_by_name("Raysid"))
    except Exception as e:
        print("Scan failed:", e)
        sys.exit(1)

    if not device:
        print("Device not found")
        sys.exit(1)

    gui = SpectrumGUI(device.address)
    plt.show()
