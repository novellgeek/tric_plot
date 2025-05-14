
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess

def run_script(master_path, compare_paths):
    try:
        # Assuming ric_compare.py is modified to accept command line args
        cmd = ["python", "ric_compare.py", master_path] + compare_paths
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", "RIC comparison completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def select_master():
    file_path = filedialog.askopenfilename(filetypes=[("TLE files", "*.txt")])
    if file_path:
        master_entry.delete(0, tk.END)
        master_entry.insert(0, file_path)

def select_comparisons():
    files = filedialog.askopenfilenames(filetypes=[("TLE files", "*.txt")])
    if files:
        compare_entry.delete(0, tk.END)
        compare_entry.insert(0, ";".join(files))

def start_analysis():
    master = master_entry.get()
    comparisons = compare_entry.get().split(";")
    if not os.path.isfile(master) or not all(os.path.isfile(f) for f in comparisons):
        messagebox.showerror("Error", "Invalid file paths provided.")
        return
    run_script(master, comparisons)

# GUI Setup
root = tk.Tk()
root.title("RIC Compare Tool")

tk.Label(root, text="Master TLE File:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
master_entry = tk.Entry(root, width=60)
master_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_master).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Comparison TLE Files:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
compare_entry = tk.Entry(root, width=60)
compare_entry.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_comparisons).grid(row=1, column=2, padx=10, pady=5)

tk.Button(root, text="Run RIC Comparison", command=start_analysis).grid(row=2, column=1, pady=20)

root.mainloop()
