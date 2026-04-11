"""
universitybox.survey._gui
==========================
Optional Tkinter GUI for the SurveySynthesizer.
No external dependencies — uses only Python stdlib (tkinter, csv, io).

Three tabs:
  Tab 1 — Schema Builder   : define questions (name, type, scale/categories)
  Tab 2 — Data Input        : load real responses from CSV
  Tab 3 — Synthesize & Export: set N, run synthesis, preview, export CSV

Launch with:
    from universitybox.survey import launch_gui
    launch_gui()
"""
from __future__ import annotations

import csv
import io
import threading
from typing import List, Optional


def launch_gui() -> None:
    """Open the Survey Synthesizer GUI window (blocks until closed)."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
    except ImportError:
        raise ImportError(
            "tkinter is required for the GUI. It is part of the Python standard "
            "library but may be missing on some systems (e.g. install python3-tk "
            "on Debian/Ubuntu)."
        )

    # ------------------------------------------------------------------ #
    # State shared across tabs
    # ------------------------------------------------------------------ #
    schema_rows: List[dict] = []   # list of {name, type, detail}
    real_df = None                  # pandas DataFrame once loaded
    result_df = None                # synthesised DataFrame

    # ------------------------------------------------------------------ #
    # Root window
    # ------------------------------------------------------------------ #
    root = tk.Tk()
    root.title("UniversityBox — Survey Synthesizer")
    root.geometry("820x620")
    root.resizable(True, True)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    # ==================================================================
    # TAB 1 — Schema Builder
    # ==================================================================
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="  Schema Builder  ")

    header = ttk.Frame(tab1)
    header.pack(fill="x", padx=8, pady=(8, 4))
    ttk.Label(header, text="Define your survey questions:", font=("", 11, "bold")).pack(side="left")

    # Question list frame (scrollable)
    list_outer = ttk.LabelFrame(tab1, text="Questions")
    list_outer.pack(fill="both", expand=True, padx=8, pady=4)

    canvas = tk.Canvas(list_outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner_frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bounding_box("all"))

    inner_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

    row_widgets: List[dict] = []

    def _refresh_rows():
        for w in inner_frame.winfo_children():
            w.destroy()
        row_widgets.clear()
        for i, row in enumerate(schema_rows):
            f = ttk.Frame(inner_frame)
            f.pack(fill="x", pady=2, padx=4)

            ttk.Label(f, text=f"{i+1}.", width=3).pack(side="left")

            name_var = tk.StringVar(value=row["name"])
            name_entry = ttk.Entry(f, textvariable=name_var, width=20)
            name_entry.pack(side="left", padx=(2, 6))

            type_var = tk.StringVar(value=row["type"])
            type_cb = ttk.Combobox(f, textvariable=type_var,
                                   values=["ordinal", "categorical", "continuous"],
                                   width=12, state="readonly")
            type_cb.pack(side="left", padx=(0, 6))

            detail_var = tk.StringVar(value=row["detail"])
            detail_entry = ttk.Entry(f, textvariable=detail_var, width=28)
            detail_entry.pack(side="left", padx=(0, 6))

            ttk.Label(f, text="← scale (1,5) / categories / bounds (0,100)",
                      foreground="gray", font=("", 8)).pack(side="left")

            def _del(idx=i):
                schema_rows.pop(idx)
                _refresh_rows()

            ttk.Button(f, text="✕", width=3, command=_del).pack(side="right")

            # Keep references so we can read later
            row_data = {"name": name_var, "type": type_var, "detail": detail_var}
            row_widgets.append(row_data)

    def _add_row():
        schema_rows.append({"name": f"Q{len(schema_rows)+1}", "type": "ordinal", "detail": "(1, 5)"})
        _refresh_rows()

    def _collect_schema():
        """Read current widget values back into schema_rows."""
        for i, rw in enumerate(row_widgets):
            schema_rows[i]["name"] = rw["name"].get().strip()
            schema_rows[i]["type"] = rw["type"].get().strip()
            schema_rows[i]["detail"] = rw["detail"].get().strip()

    btn_frame = ttk.Frame(tab1)
    btn_frame.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(btn_frame, text="+ Add Question", command=_add_row).pack(side="left", padx=4)

    # Add a couple of starter rows
    _add_row()
    _add_row()

    # ==================================================================
    # TAB 2 — Data Input
    # ==================================================================
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="  Data Input  ")

    ttk.Label(tab2, text="Load real survey responses (CSV):", font=("", 11, "bold")).pack(
        anchor="w", padx=8, pady=(8, 4)
    )

    file_var = tk.StringVar(value="No file selected")
    file_frame = ttk.Frame(tab2)
    file_frame.pack(fill="x", padx=8)
    ttk.Label(file_frame, textvariable=file_var, foreground="gray", width=55).pack(side="left")

    preview_text = scrolledtext.ScrolledText(tab2, height=18, font=("Courier", 9))
    preview_text.pack(fill="both", expand=True, padx=8, pady=4)

    status2 = tk.StringVar(value="")
    ttk.Label(tab2, textvariable=status2, foreground="green").pack(anchor="w", padx=8)

    def _load_csv():
        nonlocal real_df
        path = filedialog.askopenfilename(
            title="Select CSV with real responses",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            real_df = df
            file_var.set(path)
            # Preview
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", df.head(10).to_string(index=False))
            status2.set(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    ttk.Button(file_frame, text="Browse…", command=_load_csv).pack(side="left", padx=8)

    # ==================================================================
    # TAB 3 — Synthesize & Export
    # ==================================================================
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="  Synthesize & Export  ")

    settings_frame = ttk.LabelFrame(tab3, text="Settings")
    settings_frame.pack(fill="x", padx=8, pady=8)

    def _slider_row(parent, label, var, from_, to, default, row):
        ttk.Label(parent, text=label, width=24, anchor="w").grid(
            row=row, column=0, sticky="w", padx=6, pady=3
        )
        sl = ttk.Scale(parent, variable=var, from_=from_, to=to, orient="horizontal", length=220)
        sl.grid(row=row, column=1, padx=6)
        val_lbl = ttk.Label(parent, textvariable=var, width=8)
        val_lbl.grid(row=row, column=2, padx=4)
        var.set(default)

    n_var = tk.IntVar(value=1000)
    mcmc_var = tk.IntVar(value=500)
    seeds_var = tk.IntVar(value=10)
    nhop_var = tk.DoubleVar(value=99.0)

    _slider_row(settings_frame, "Synthetic N", n_var, 100, 10000, 1000, 0)
    _slider_row(settings_frame, "MCMC iterations", mcmc_var, 100, 2000, 500, 1)
    _slider_row(settings_frame, "Seeds (k-means++)", seeds_var, 1, 50, 10, 2)
    _slider_row(settings_frame, "NHOP percentile", nhop_var, 50, 100, 99, 3)

    run_btn = ttk.Button(tab3, text="▶  Run Synthesis")
    run_btn.pack(pady=6)

    progress = ttk.Progressbar(tab3, mode="indeterminate", length=300)
    progress.pack(pady=4)

    status3 = tk.StringVar(value="")
    ttk.Label(tab3, textvariable=status3, foreground="blue").pack()

    preview3 = scrolledtext.ScrolledText(tab3, height=12, font=("Courier", 9))
    preview3.pack(fill="both", expand=True, padx=8, pady=4)

    export_btn = ttk.Button(tab3, text="Export to CSV…", state="disabled")
    export_btn.pack(pady=4)

    def _do_synthesis():
        nonlocal result_df
        _collect_schema()

        if real_df is None:
            messagebox.showwarning("No data", "Please load real responses in Tab 2 first.")
            return

        # Build schema
        try:
            from ._schema import SurveySchema
            schema = SurveySchema()
            for row in schema_rows:
                name = row["name"]
                qtype = row["type"]
                detail = row["detail"].strip()
                if qtype == "categorical":
                    cats = [c.strip().strip("'\"") for c in detail.strip("[]()").split(",")]
                    schema.add_categorical(name, categories=cats)
                elif qtype == "ordinal":
                    parts = detail.strip("()[]").split(",")
                    lo, hi = int(parts[0].strip()), int(parts[1].strip())
                    schema.add_ordinal(name, scale=(lo, hi))
                else:
                    parts = detail.strip("()[]").split(",")
                    lo, hi = float(parts[0].strip()), float(parts[1].strip())
                    schema.add_continuous(name, bounds=(lo, hi))
        except Exception as exc:
            messagebox.showerror("Schema error", f"Check schema settings:\n{exc}")
            return

        run_btn.config(state="disabled")
        progress.start(10)
        status3.set("Running…")

        def _run():
            nonlocal result_df
            try:
                from ._synthesizer import SurveySynthesizer
                synth = SurveySynthesizer(
                    n_mcmc=int(mcmc_var.get()),
                    n_seeds=int(seeds_var.get()),
                    nhop_pct=float(nhop_var.get()),
                )
                synth.fit(real_df, schema)
                result_df = synth.synthesize(N=int(n_var.get()))
                root.after(0, _done, None)
            except Exception as exc:
                root.after(0, _done, str(exc))

        def _done(err):
            progress.stop()
            run_btn.config(state="normal")
            if err:
                status3.set(f"Error: {err}")
                messagebox.showerror("Synthesis failed", err)
            else:
                status3.set(f"Done — {len(result_df)} synthetic responses generated.")
                preview3.delete("1.0", "end")
                preview3.insert("1.0", result_df.head(10).to_string(index=False))
                export_btn.config(state="normal")

        threading.Thread(target=_run, daemon=True).start()

    run_btn.config(command=_do_synthesis)

    def _export():
        if result_df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save synthetic responses",
        )
        if path:
            result_df.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"Saved {len(result_df)} rows to:\n{path}")

    export_btn.config(command=_export)

    root.mainloop()
