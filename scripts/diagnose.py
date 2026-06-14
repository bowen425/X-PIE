#!/usr/bin/env python3
import subprocess
import os
import platform

print("=" * 60)
print("SYSTEM: " + platform.platform())
print("PYTHON: " + os.popen("python3 --version").read().strip())
print("=" * 60)

# 1. Check mkdssp
print("\n--- 1. DSSP Version ---")
result = subprocess.run(["mkdssp", "--version"], capture_output=True, text=True)
print("stdout: " + result.stdout[:200])
print("stderr: " + result.stderr[:200])
print("rc: " + str(result.returncode))

# 2. Check Python packages
print("\n--- 2. Package Versions ---")
for mod in ["numpy", "matplotlib", "Bio"]:
    try:
        m = __import__(mod)
        print(mod + ": " + getattr(m, "__version__", "unknown"))
    except:
        print(mod + ": NOT FOUND")

# 3. Test DSSP on your PDB
print("\n--- 3. DSSP Functional Test ---")
test_pdb = """ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00 72.59           N
ATOM      2  CA  ALA A   1      27.340  25.720   3.212  1.00 72.59           C
ATOM      3  C   ALA A   1      26.266  26.639   2.669  1.00 72.59           C
ATOM      4  O   ALA A   1      25.252  26.467   3.328  1.00 72.59           O
ATOM      5  CB  ALA A   1      27.341  25.634   4.722  1.00 72.59           C
END
"""
with open("/tmp/test.pdb", "w") as f:
    f.write(test_pdb)

from Bio.PDB import PDBParser, DSSP
parser = PDBParser(QUIET=True)
structure = parser.get_structure("test", "/tmp/test.pdb")
model = list(structure.get_models())[0]

try:
    dssp = DSSP(model, "/tmp/test.pdb", dssp="mkdssp")
    print("Standard DSSP: OK, residues=" + str(len(list(dssp.keys()))))
except Exception as e:
    print("Standard DSSP FAILED: " + str(e)[:100])
    try:
        dssp = DSSP(model, "/tmp/test.pdb", dssp="mkdssp", file_type="PDB")
        print("With file_type='PDB': OK")
    except Exception as e2:
        print("With file_type='PDB' ALSO FAILED: " + str(e2)[:100])

# 4. Test command line mkdssp
print("\n--- 4. Command Line mkdssp ---")
for cmd in [["mkdssp", "-i", "/tmp/test.pdb", "-o", "/tmp/test.dssp"],
            ["mkdssp", "/tmp/test.pdb", "/tmp/test.dssp"]]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists("/tmp/test.dssp") and os.path.getsize("/tmp/test.dssp") > 0:
        print("Command " + " ".join(cmd) + ": OK")
        os.remove("/tmp/test.dssp")
        break
    else:
        print("Command " + " ".join(cmd) + ": FAILED rc=" + str(result.returncode))

# 5. Matplotlib figure size test
print("\n--- 5. Matplotlib Figure Size ---")
import matplotlib.pyplot as plt
for w in [12, 50, 100, 200]:
    try:
        fig = plt.figure(figsize=(w, 20))
        fig.savefig("/tmp/testfig.png", dpi=300)
        plt.close(fig)
        os.remove("/tmp/testfig.png")
        print("figsize=(" + str(w) + ", 20) @ 300dpi: OK")
    except Exception as e:
        print("figsize=(" + str(w) + ", 20): FAILED - " + str(e)[:80])
        break

print("\n" + "=" * 60)
print("Done. Please copy all output above.")
print("=" * 60)
