from __future__ import annotations
import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit
)

import omnipose_threaded
import process_masks

# Constants similar to simple_omnipose_gui
MAX_WORKERS = os.cpu_count() // 2
PLATE_TYPE = "96W"
MAGNIFICATION = "10x"
CYTATION = True
CHANNEL_ORDER = [1, 2, 0]

PLATE_AREAS = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON = 1 / 10000
if CYTATION:
    MICRONS_PER_PIXEL = 1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992
    IMAGE_AREA_CM = 1992 * 1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2
else:
    if MAGNIFICATION == "10x":
        MICRONS_PER_PIXEL = 0.61922571983322461
        IMAGE_AREA_CM = 0.0120619953
    else:
        MICRONS_PER_PIXEL = 1.5188172690164046
        IMAGE_AREA_CM = 0.0725658405

CM_PER_PIXEL = CM_PER_MICRON * MICRONS_PER_PIXEL


class OmniposeGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Omnipose Pipeline (Qt)")
        self.resize(600, 400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Input directory
        inp_layout = QHBoxLayout()
        inp_label = QLabel("Input directory:")
        self.inp_edit = QLineEdit()
        browse_inp = QPushButton("Browse")
        browse_inp.clicked.connect(self._browse_input)
        inp_layout.addWidget(inp_label)
        inp_layout.addWidget(self.inp_edit)
        inp_layout.addWidget(browse_inp)
        layout.addLayout(inp_layout)

        # Model file
        mod_layout = QHBoxLayout()
        mod_label = QLabel("Model file:")
        self.mod_edit = QLineEdit()
        browse_mod = QPushButton("Browse")
        browse_mod.clicked.connect(self._browse_model)
        mod_layout.addWidget(mod_label)
        mod_layout.addWidget(self.mod_edit)
        mod_layout.addWidget(browse_mod)
        layout.addLayout(mod_layout)

        # Filtering keyword
        filt_layout = QHBoxLayout()
        filt_label = QLabel("Filter keyword:")
        self.filt_edit = QLineEdit("bright")
        filt_layout.addWidget(filt_label)
        filt_layout.addWidget(self.filt_edit)
        layout.addLayout(filt_layout)

        # Z indices
        z_layout = QHBoxLayout()
        z_label = QLabel("Z indices (comma separated):")
        self.z_edit = QLineEdit("0,1,2")
        z_layout.addWidget(z_label)
        z_layout.addWidget(self.z_edit)
        layout.addLayout(z_layout)

        # Options
        opt_layout = QHBoxLayout()
        self.cb_flows = QCheckBox("Save flows")
        self.cb_cellprob = QCheckBox("Save cell prob")
        self.cb_outlines = QCheckBox("Save outlines")
        self.cb_outlines.setChecked(True)
        opt_layout.addWidget(self.cb_flows)
        opt_layout.addWidget(self.cb_cellprob)
        opt_layout.addWidget(self.cb_outlines)
        layout.addLayout(opt_layout)

        run_btn = QPushButton("Run Omnipose")
        run_btn.clicked.connect(self._run_pipeline)
        layout.addWidget(run_btn)

        # --- Run R section ---
        csv_layout = QHBoxLayout()
        csv_label = QLabel("all_data_csv:")
        self.csv_edit = QLineEdit()
        browse_csv = QPushButton("Browse")
        browse_csv.clicked.connect(self._browse_csv)
        csv_layout.addWidget(csv_label)
        csv_layout.addWidget(self.csv_edit)
        csv_layout.addWidget(browse_csv)
        layout.addLayout(csv_layout)

        r_btn = QPushButton("Run R Analysis")
        r_btn.clicked.connect(self._run_rscript)
        layout.addWidget(r_btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    # ----- helpers -----
    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            self.inp_edit.setText(path)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "omnipose_models")
        if path:
            self.mod_edit.setText(path)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select all_data_csv", filter="CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def _append_log(self, text: str) -> None:
        self.log.append(text)
        self.log.ensureCursorVisible()

    def _run_pipeline(self) -> None:
        input_dir = self.inp_edit.text()
        model_file = self.mod_edit.text()
        if not Path(input_dir).is_dir():
            self._append_log("Invalid input directory")
            return
        if not Path(model_file).is_file():
            self._append_log("Model file not found")
            return

        filt = self.filt_edit.text().strip() or "bright"
        z_text = self.z_edit.text().strip()
        z_indices = [int(i) for i in z_text.split(',') if i.strip().isdigit()] if z_text else None

        self._append_log("Running Omnipose...")
        model_info = (model_file, 0.4, 0)
        for sub in os.listdir(input_dir):
            sub_path = os.path.join(input_dir, sub)
            if not os.path.isdir(sub_path) or "epoch" in sub:
                continue
            out_dir = omnipose_threaded.run_omnipose(
                sub_path,
                model_info,
                num_threads=MAX_WORKERS,
                channel_order=CHANNEL_ORDER,
                filter_keyword=filt,
                z_indices=z_indices,
                save_flows=self.cb_flows.isChecked(),
                save_cellProb=self.cb_cellprob.isChecked(),
                save_outlines=self.cb_outlines.isChecked(),
            )
            process_masks.process_mask_files(
                out_dir,
                CM_PER_PIXEL,
                PLATE_AREAS.get(PLATE_TYPE),
                force_save=False,
                filter_min_size=None,
            )
        all_csv = process_masks.make_all_data_csv(input_dir, os.path.basename(model_info[0]))
        if all_csv:
            self.csv_edit.setText(all_csv)
            self._append_log(f"Created {all_csv}")
        self._append_log("Finished Omnipose pipeline")

    def _run_rscript(self) -> None:
        csv_path = self.csv_edit.text()
        if not Path(csv_path).is_file():
            self._append_log("Select a valid CSV file")
            return
        script = Path(__file__).with_name("interaction.R")
        cmd = ["Rscript", str(script), csv_path]
        self._append_log("Running Rscript interact.R ...")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._append_log(out.stdout)
            if out.stderr:
                self._append_log(out.stderr)
            self._append_log("R script finished")
        except subprocess.CalledProcessError as e:
            self._append_log(f"R script failed: {e}\n{e.stdout}\n{e.stderr}")


def main() -> None:
    app = QApplication([])
    gui = OmniposeGUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()

