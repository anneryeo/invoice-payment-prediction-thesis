import json, glob, os
for f in glob.glob("**/*.ipynb", recursive=True):
    # Skip checkpoint files
    if ".ipynb_checkpoints" in f: continue
    try:
        with open(f, "r", encoding="utf-8") as file:
            nb = json.load(file)
        code = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                s = "".join(cell["source"])
                if s.strip():
                    if not s.endswith("\n"): s += "\n"
                    code.append(s)
                    code.append("\n")
            elif cell["cell_type"] == "markdown":
                s = "".join(cell["source"])
                if s.strip():
                    code.append("# " + "\n# ".join(s.splitlines()) + "\n\n")
        
        py_f = f.replace(".ipynb", ".py")
        with open(py_f, "w", encoding="utf-8") as file:
            file.write("".join(code))
        print(f"Converted: {f} -> {py_f}")
    except Exception as e:
        print(f"Failed: {f} - {e}")
