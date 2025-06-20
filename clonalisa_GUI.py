from __future__ import annotations
import os, sys, subprocess
from pathlib import Path

import omnipose_threaded, process_masks
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem,
    QInputDialog, QDialog, QFormLayout, QLineEdit, QDialogButtonBox
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Signal, QThread, QEvent
from PySide6.QtUiTools import QUiLoader

from config_utils import load_config, save_config, parse_filename

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore    import QFile, QObject

from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QHeaderView

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QStyleFactory

from colorsys import hsv_to_rgb
import matplotlib.cm      as cm
import matplotlib.colors  as mcolors

def enable_dark_palette(app):
    """Switch to Fusion style + dark palette."""
    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()

    # basic tones
    dark   = QColor(45, 45, 45)
    mid    = QColor(60, 60, 60)
    light  = QColor(90, 90, 90)
    text   = QColor(220, 220, 220)
    accent = QColor(53, 132, 228)  # links / highlights

    pal.setColor(QPalette.Window,        dark)
    pal.setColor(QPalette.WindowText,    text)
    pal.setColor(QPalette.Base,          mid)
    pal.setColor(QPalette.AlternateBase, dark)
    pal.setColor(QPalette.ToolTipBase,   text)
    pal.setColor(QPalette.ToolTipText,   text)
    pal.setColor(QPalette.Text,          text)
    pal.setColor(QPalette.Button,        mid)
    pal.setColor(QPalette.ButtonText,    text)
    pal.setColor(QPalette.Link,          accent)
    pal.setColor(QPalette.Highlight,     accent)
    pal.setColor(QPalette.HighlightedText, QColor("white"))

    app.setPalette(pal)


# -------- constants ----------------------------------------------------------
MAX_WORKERS   = os.cpu_count() // 2
PLATE_TYPE    = "96W"
MAGNIFICATION = "10x"
CYTATION      = True

PLATE_AREAS        = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON      = 1 / 10000
MICRONS_PER_PIXEL  = (1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992) if CYTATION \
                     else (0.61922571983322461 if MAGNIFICATION == "10x" else 1.5188172690164046)
CM_PER_PIXEL       = CM_PER_MICRON * MICRONS_PER_PIXEL
# -----------------------------------------------------------------------------


class ConfigDialog(QDialog):
    """Simple dialog for editing regex settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.resize(800, 50)
        self.cfg = load_config()

        layout = QFormLayout(self)
        self.regex_file = QLineEdit(self.cfg.get('regex', {}).get('file', ''))
        self.regex_time = QLineEdit(self.cfg.get('regex', {}).get('time_from_folder', ''))
        layout.addRow("Filename regex", self.regex_file)
        layout.addRow("Time regex",      self.regex_time)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def save(self):
        self.cfg['regex'] = {
            'file': self.regex_file.text(),
            'time_from_folder': self.regex_time.text(),
        }
        save_config(self.cfg)

def load_ui(path: str | os.PathLike, container: QWidget):
    loader = QUiLoader()

    # ---- load the form WITHOUT passing the container --------------
    ui_file = QFile(str(path))
    ui_file.open(QFile.ReadOnly)
    form = loader.load(ui_file)          # a QWidget with your splitter etc.
    ui_file.close()

    # ---- give the window a layout and shove the form in -----------
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(form)

    # ---- expose children as attributes (optional) -----------------
    for obj in form.findChildren(QObject):
        if obj.objectName():
            setattr(container, obj.objectName(), obj)


class PipelineWorker(QThread):
    progress_overall = Signal(int, int)
    progress_subdir = Signal(int, int)
    log = Signal(str)
    csv_created = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        input_dir = Path(self.params['input_dir'])
        model_info = self.params['model_info']
        subdirs = [d for d in input_dir.iterdir() if d.is_dir() and 'epoch' not in d.name]
        total_dirs = len(subdirs)
        for idx, sub in enumerate(subdirs, start=1):
            self.log.emit(f"Processing {sub.name} ({idx}/{total_dirs})")

            def cb(done, total):
                self.progress_subdir.emit(done, total)

            out_dir = omnipose_threaded.run_omnipose(
                sub,
                model_info,
                filter_keyword=self.params['filt'],
                z_indices=self.params['z_indices'],
                save_cellProb=self.params['save_cellProb'],
                save_flows=self.params['save_flows'],
                save_outlines=self.params['save_outlines'],
                progress_callback=cb,
            )
            process_masks.process_mask_files(
                out_dir,
                CM_PER_PIXEL,
                PLATE_AREAS.get(PLATE_TYPE),
                force_save=False,
                filter_min_size=None,
            )
            self.progress_overall.emit(idx, total_dirs)

        all_csv = process_masks.make_all_data_csv(str(input_dir), Path(model_info[0]).name)
        if all_csv:
            self.csv_created.emit(all_csv)
            self.log.emit(f"Created {all_csv}")
        self.log.emit("Finished Omnipose pipeline")
                    
class ClonaLiSAGUI(QWidget):
    """Main window – UI is loaded from clonalisa.ui."""
    def __init__(self):
        super().__init__()

        # ----- load the .ui file ------------------------------------------------
        ui_path = Path(__file__).with_name("clonalisa_ui.ui")
        load_ui(ui_path, self)  
        self.resize(1600, 900)
        # -----------------------------------------------------------------------

        # internal state --------------------------------------------------------
        self.cfg         = load_config()
        self.plate_wells = {}   # { plate: {A1, B3, ...} }
        self.group_data  = {}   # { plate: {group: {well: value}} }
        self.cell_data   = {}   # { plate: {timepoint: {well: density}} }
        self.timepoints  = []   # sorted list of timepoints for current plate
        # -----------------------------------------------------------------------

        # ----- post-load tweaks Designer can’t express -----------------------
        self.table: QTableWidget
        self.table.setRowCount(8)
        self.table.setColumnCount(12)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setHorizontalHeaderLabels([str(i + 1) for i in range(12)])
        self.table.setVerticalHeaderLabels([chr(ord('A') + i) for i in range(8)])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.installEventFilter(self)

        self.value_colors = {}          # { "treatmentA": QColor, ... }
        self._next_hue    = 0           # rolling hue pointer (0-359)
        self._cmap = cm.get_cmap("viridis")

        # slider for timepoints
        self.time_slider.setVisible(False)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._update_grid)

        self.progressSubdir.setValue(0)
        self.progressOverall.setValue(0)
        self._update_fixed_effect_options()

        # self.mainSplitter.setSizes([0, 1])    # "1" = take the rest

        # ---------------------------------------------------------------------

        # ---------- connect signals ------------------------------------------
        self.btnBrowseInput.clicked.connect(self._browse_input)
        self.btnBrowseModel.clicked.connect(self._browse_model)
        self.btnBrowseCSV.clicked.connect(self._browse_csv)

        self.plate_combo.currentIndexChanged.connect(self._plate_selected)
        self.view_combo.currentIndexChanged.connect(self._update_grid)
        self.model_combo.currentIndexChanged.connect(self._model_selected)

        self.btnNewGroup.clicked.connect(self._new_group)
        self.btnApplyGroup.clicked.connect(self._apply_group)
        self.btnSaveGroups.clicked.connect(self._save_group_map)

        self.btnRunOmnipose.clicked.connect(self._run_pipeline)
        self.btnRunR.clicked.connect(self._run_rscript)
        self.btnConfig.clicked.connect(self._open_config)
        self.fe2_combo.currentIndexChanged.connect(self._update_ref_levels)
        self.plate_combo.currentIndexChanged.connect(self._update_ref_levels)
        self.csv_edit.textChanged.connect(self._update_fixed_effect_options)
        self.csv_edit.textChanged.connect(self._load_cell_density_data)
        # ---------------------------------------------------------------------

        # internal state, preload model history …
        self.cfg         = load_config()
        self.plate_wells = {}
        self.group_data  = {}
        self.cell_data   = {}
        self.timepoints  = []
        self._model_selected(0)
        self._load_cell_density_data()

    # ==========================================================================
    # ---------------------------- helper slots --------------------------------
    # ==========================================================================

    def _color_for_value(self, val: str) -> QColor:
        """Assign (or retrieve) a pastel color for a specific value string."""
        if val not in self.value_colors:
            # step through hues 40° apart for variety
            h = (self._next_hue % 360) / 360.0
            self._next_hue += 40
            r, g, b = hsv_to_rgb(h, 0.45, 0.85)  # pastel-ish
            self.value_colors[val] = QColor(int(r*255), int(g*255), int(b*255))
        return self.value_colors[val]

    # -- generic --------------------------------------------------------------
    def _append_log(self, text: str):
        self.log.append(text)
        self.log.ensureCursorVisible()

    def _update_subdir_progress(self, done: int, total: int):
        self.progressSubdir.setMaximum(total)
        self.progressSubdir.setValue(done)

    def _update_overall_progress(self, done: int, total: int):
        self.progressOverall.setMaximum(total)
        self.progressOverall.setValue(done)

    def _expected_csv_path(self) -> Path | None:
        inp = Path(self.inp_edit.text())
        model = Path(self.model_combo.currentText())
        if inp.is_dir() and model.name:
            return inp / model.name / f"{model.name}_all_data.csv"
        return None

    def _set_expected_csv(self):
        exp = self._expected_csv_path()
        if exp and exp.is_file():
            self.csv_edit.setText(str(exp))
        self._update_fixed_effect_options()

    def _pipeline_finished(self, model_file: Path, flow_thr: float, mask_thr: float, z_indices):
        entry = {
            'path': str(model_file), 'flow_threshold': flow_thr,
            'mask_threshold': mask_thr, 'z_indices': z_indices or [],
        }
        self.cfg['model_history'] = [entry] + [
            e for e in self.cfg.get('model_history', []) if e['path'] != str(model_file)
        ]
        save_config(self.cfg)
        self._refresh_model_combo()

    # -- browse helpers --------------------------------------------------------
    def _browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            self.inp_edit.setText(path)
            self._load_plates(path)
            self._set_expected_csv()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "omnipose_models")
        if path and path not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.addItem(path)
        self.model_combo.setCurrentText(path)
        self._set_expected_csv()

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select all_data_csv", filter="CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    # -- model-history ---------------------------------------------------------
    def _model_selected(self, idx: int):
        hist = self.cfg.get('model_history', [])
        if 0 <= idx < len(hist):
            entry = hist[idx]
            self.model_combo.setEditText(entry['path'])
            self.flow_edit.setText(str(entry.get('flow_threshold', '')))
            self.mask_edit.setText(str(entry.get('mask_threshold', '')))
            if entry.get('z_indices'):
                self.z_edit.setText(','.join(map(str, entry['z_indices'])))
        self._set_expected_csv()

    def _refresh_model_combo(self):
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for entry in self.cfg.get('model_history', []):
            self.model_combo.addItem(entry['path'])
        self.model_combo.blockSignals(False)
        self.model_combo.setCurrentText(current)
        self._set_expected_csv()

    def _load_cell_density_data(self):
        """Load and average cell_density per Plate/Timepoint/Well."""
        self.cell_data.clear()
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            self._update_slider()
            return

        try:
            df = pd.read_csv(csv_path,
                            usecols=['Plate', 'Well',
                                    'Relative Time (hrs)', 'cell_density'])
        except Exception:
            self._update_slider()
            return

        df['Plate'] = df['Plate'].astype(str).str.lower()

        # average cell_density for identical Plate-Time-Well combos
        grouped = (
            df.groupby(['Plate', 'Relative Time (hrs)', 'Well'],
                    as_index=False)['cell_density']
            .mean()
        )

        for (plate, t), grp in grouped.groupby(['Plate', 'Relative Time (hrs)']):
            mapping = dict(zip(grp['Well'], grp['cell_density']))
            self.cell_data.setdefault(plate, {})[t] = mapping

        self._update_slider()

    def _update_slider(self):
        """Configure the time slider for the currently selected plate."""
        plate = self.plate_combo.currentText()
        times = sorted(self.cell_data.get(plate, {}).keys())
        self.timepoints = times
        if len(times) > 1 and self.view_combo.currentText() == 'Cell Density':
            self.time_slider.setVisible(True)
            self.time_slider.setMaximum(len(times) - 1)
            if self.time_slider.value() > len(times) - 1:
                self.time_slider.setValue(0)
        else:
            self.time_slider.setVisible(False)
            self.time_slider.setMaximum(0)
            self.time_slider.setValue(0)

    # ------------------------------------------------------------------
    # Plate / well grid -------------------------------------------------
    # ------------------------------------------------------------------
    def _load_plates(self, folder: str):
        """
        Populate plate list, view list, well sets, and group mappings.

        * Plates are discovered in two ways:
            1. Sub-folders in the chosen input directory.
            2. The Plate column in group_map.csv (if it exists).
        """
        # Avoid cascade of currentIndexChanged slots while we rebuild
        self.plate_combo.blockSignals(True)
        self.view_combo.blockSignals(True)

        # ---------- full reset ---------------------------------------
        self.plate_combo.clear()
        self.view_combo.clear()
        self.view_combo.addItem("Imaged Wells")
        self.view_combo.addItem("Cell Density")
        self.table.clearContents()
        self.plate_wells.clear()
        self.group_data.clear()

        # ---------- 1) discover plates from sub-folders --------------
        discovered_plates = {
            sub.split('_')[0]
            for sub in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, sub))
        }

        # ---------- 2) optionally load group_map.csv ----------------
        csv_path = Path(folder) / "group_map.csv"
        if csv_path.is_file():
            df = pd.read_csv(csv_path).fillna("")
            df.columns = [c.strip() for c in df.columns]

            for _, row in df.iterrows():
                plate = str(row["Plate"]).strip()     # <- NO NORMALISATION
                well  = row["Well"].strip()

                # ensure plate shows up in GUI even if not on disk
                self.plate_wells.setdefault(plate, set()).add(well)

                for group_name, val in row.items():
                    if group_name in ("Plate", "Well") or val == "":
                        continue

                    (
                        self.group_data
                        .setdefault(plate, {})
                        .setdefault(group_name, {})
                    )[well] = str(val)

                    if self.view_combo.findText(group_name) == -1:
                        self.view_combo.addItem(group_name)

        # ---------- 3) merge folder-found plates --------------------
        for plate in sorted(discovered_plates):
            if plate not in self.plate_wells:
                self.plate_wells[plate] = set()        # wells will fill later

        # ---------- 4) populate plate_combo -------------------------
        for plate in sorted(self.plate_wells.keys()):
            self.plate_combo.addItem(plate)

        # ---------- 5) wrap-up --------------------------------------
        self.plate_combo.blockSignals(False)
        self.view_combo.blockSignals(False)

        # update fixed-effect drop-downs and show first plate
        self._update_fixed_effect_options()
        if self.plate_combo.count() > 0:
            self.plate_combo.setCurrentIndex(0)   # triggers _plate_selected
            self._plate_selected(None)


    def _plate_selected(self, _):
        plate = self.plate_combo.currentText()
        if not plate:
            return
        input_dir = self.inp_edit.text()
        wells = set()
        for sub in os.listdir(input_dir):
            if not sub.startswith(plate):
                continue
            for root, _, files in os.walk(os.path.join(input_dir, sub)):
                for f in files:
                    if f.lower().endswith('.tif'):
                        well, *_ = parse_filename(f)
                        if well:
                            wells.add(well)
                break
        self.plate_wells[plate] = wells
        self._update_slider()
        self._update_grid()

    def _index_to_well(self, row: int, col: int) -> str:
        return f"{chr(ord('A') + row)}{col + 1}"

    def _update_grid(self):
        plate = self.plate_combo.currentText()
        if not plate:
            return

        view  = self.view_combo.currentText()
        wells = self.plate_wells.get(plate, set())
        self._update_slider()

        for r in range(8):
            for c in range(12):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)

                well = self._index_to_well(r, c)
                item.setText("")                          

                if view == "Imaged Wells":
                    item.setBackground(
                        QColor("lightgreen" if well in wells else "lightgray")
                    )
                elif view == "Cell Density":
                    if not self.timepoints:
                        mapping = {}
                    else:
                        idx   = min(self.time_slider.value(), len(self.timepoints) - 1)
                        time  = self.timepoints[idx]
                        mapping = self.cell_data.get(plate, {}).get(time, {})

                    # normalise once for the whole plate/timepoint
                    if mapping:
                        vmin = min(mapping.values())
                        vmax = max(mapping.values())
                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    else:
                        norm = None

                    val = mapping.get(well)
                    if val is not None and norm is not None:
                        rgba   = self._cmap(norm(val))
                        red, green, blue = (int(round(c*255)) for c in rgba[:3])
                        item.setBackground(QColor(red, green, blue))

                        # text & contrast
                        luminance = 0.299*red + 0.587*green + 0.114*blue
                        text_col  = QColor("black" if luminance > 128 else "white")
                        item.setForeground(text_col)
                        item.setText(f"{val:.2f}")
                    else:
                        item.setBackground(QColor("transparent"))
                        item.setText("")
                else:
                    mapping = self.group_data.get(plate, {}).get(view, {})
                    val     = mapping.get(well)
                    if val is not None:
                        bg = self._color_for_value(val)
                        item.setBackground(bg)

                        # pick black/white text for readability
                        luminance = 0.299*bg.redF() + 0.587*bg.greenF() + 0.114*bg.blueF()
                        fg = QColor("black" if luminance > 0.6 else "white")
                        item.setForeground(fg)

                        item.setText(str(val))
                    else:
                        item.setBackground(QColor("transparent"))
                        item.setText("")


    # -- groups ---------------------------------------------------------------
    def _new_group(self):
        name, ok = QInputDialog.getText(self, "New Group", "Group name:")
        if ok and name and self.view_combo.findText(name) == -1:
            self.view_combo.addItem(name)
            # jump straight to the new group so the user can start tagging wells
            self.view_combo.setCurrentText(name)
            self._update_fixed_effect_options()

    def _apply_group(self):
        plate = self.plate_combo.currentText()
        group = self.view_combo.currentText()
        value = self.value_edit.text()

        # don’t let the user scribble on the default “Imaged Wells” layer
        if group == "Imaged Wells" or not (plate and group and value):
            return

        sel = self.table.selectedIndexes()
        if not sel:
            return

        mapping = self.group_data.setdefault(plate, {}).setdefault(group, {})
        for idx in sel:
            mapping[self._index_to_well(idx.row(), idx.column())] = value

        self.value_edit.clear()
        self._update_grid()
        self._update_ref_levels()

    def _save_group_map(self):
        folder = self.inp_edit.text()
        if not folder:
            return
        rows = []
        for plate, groups in self.group_data.items():
            wells = {w for g in groups.values() for w in g}
            for well in wells:
                row = {"Plate": plate, "Well": well}
                for gname, mapping in groups.items():
                    row[gname] = mapping.get(well)
                rows.append(row)
        if rows:
            csv_path = Path(folder) / "group_map.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            self._append_log("Saved group_map.csv")

            csv = Path(self.csv_edit.text())
            if csv.is_file():
                updated = process_masks.update_groups_in_all_csv(csv, csv_path)
                if updated:
                    self.csv_edit.setText(updated)
                    self._append_log("Updated all_data CSV with groups")
                    self._update_fixed_effect_options()
                    self._update_ref_levels()
            else:
                self._append_log("all_data CSV not found; run pipeline first")
        else:
            self._append_log("No groups to save")

    def _update_fixed_effect_options(self):
        csv_path = Path(self.csv_edit.text())
        if csv_path.is_file():
            try:
                df = pd.read_csv(csv_path, nrows=1)
                groups = [c[6:] for c in df.columns if c.startswith("Group-")]
            except Exception:
                groups = []
        else:
            groups = []
        self.fe1_combo.clear()
        self.fe2_combo.clear()
        self.fe1_combo.addItem("")
        self.fe2_combo.addItem("")
        self.fe1_combo.addItems(groups)
        self.fe2_combo.addItems(groups)
        self._update_ref_levels()

    def _update_ref_levels(self):
        group = self.fe2_combo.currentText()
        csv_path = Path(self.csv_edit.text())
        self.ref_level_combo.clear()
        if csv_path.is_file() and group:
            try:
                col = f"Group-{group}"
                df = pd.read_csv(csv_path, usecols=[col])
                levels = sorted(df[col].dropna().unique())
                self.ref_level_combo.addItems([str(v) for v in levels])
            except Exception:
                pass

    def eventFilter(self, obj, ev):
        if obj is self.table and ev.type() == QEvent.KeyPress and ev.key() == Qt.Key_Backspace:
            self._clear_group_selection()                       # own helper (below)
            return True                                         # swallow event
        return super().eventFilter(obj, ev)

    def _clear_group_selection(self):
        plate = self.plate_combo.currentText()
        group = self.view_combo.currentText()

        # don’t touch the default view or empty selections
        if group == "Imaged Wells" or not plate:
            return

        sel = self.table.selectedIndexes()
        if not sel:
            return

        mapping = self.group_data.get(plate, {}).get(group, {})
        for idx in sel:
            well = self._index_to_well(idx.row(), idx.column())
            mapping.pop(well, None)                              # delete if present

        # drop empty group maps to keep things tidy
        if mapping == {}:
            self.group_data[plate].pop(group, None)

        self._update_grid()
        self._update_ref_levels()

    # -- config ---------------------------------------------------------------
    def _open_config(self):
        dlg = ConfigDialog(self)
        if dlg.exec() == QDialog.Accepted:
            dlg.save()
            self.cfg = load_config()
            self._refresh_model_combo()

    # -- pipeline -------------------------------------------------------------
    def _run_pipeline(self):
        input_dir  = Path(self.inp_edit.text())
        model_file = Path(self.model_combo.currentText())
        if not input_dir.is_dir():
            self._append_log("Invalid input directory")
            return
        if not model_file.is_file():
            self._append_log("Model file not found")
            return

        filt       = self.filt_edit.text() or "bright"
        z_indices  = [int(z) for z in self.z_edit.text().split(',') if z.strip().isdigit()] \
                     if self.z_edit.text().strip() else None
        flow_thr   = float(self.flow_edit.text() or 0)
        mask_thr   = float(self.mask_edit.text() or 0)
        self._append_log("Running Omnipose…")
        params = dict(
            input_dir=str(input_dir),
            model_info=(str(model_file), flow_thr, mask_thr),
            filt=filt,
            z_indices=z_indices,
            save_cellProb=self.cb_cellprob.isChecked(),
            save_flows=self.cb_flows.isChecked(),
            save_outlines=self.cb_outlines.isChecked(),
        )
        self.worker = PipelineWorker(params)
        self.worker.progress_subdir.connect(self._update_subdir_progress)
        self.worker.progress_overall.connect(self._update_overall_progress)
        self.worker.log.connect(self._append_log)
        self.worker.csv_created.connect(lambda p: self.csv_edit.setText(p))
        self.worker.finished.connect(lambda: self._pipeline_finished(model_file, flow_thr, mask_thr, z_indices))
        self.worker.start()

    # -- R analysis -----------------------------------------------------------
    def _run_rscript(self):
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            self._append_log("Select a valid CSV file")
            return
        script = Path(__file__).with_name("interaction.R")
        self._append_log("Running Growth-Rate Analysis R script…")
        try:
            fe1 = self.fe1_combo.currentText()
            fe2 = self.fe2_combo.currentText()
            interact = "1" if self.cb_interaction.isChecked() else "0"
            ref_val = self.ref_level_combo.currentText() if self.cb_interaction.isChecked() else ""
            args = ["Rscript", str(script), str(csv_path), fe1, fe2, interact, ref_val]
            out = subprocess.run(args,
                                 capture_output=True, text=True, check=True)
            self._append_log(out.stdout)
            if out.stderr:
                self._append_log(out.stderr)
            self._append_log("R script finished")
        except subprocess.CalledProcessError as e:
            self._append_log(f"R script failed: {e}\n{e.stdout}\n{e.stderr}")


# -----------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    enable_dark_palette(app)
    gui = ClonaLiSAGUI()
    gui.show()
    gui._browse_input()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
