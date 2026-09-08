"""Shim untuk NumPy 2.x.

pyiqa menarik imgaug, dan imgaug masih membaca atribut NumPy yang DIHAPUS pada
NumPy 2.0 (`np.sctypes` dan beberapa alias dtype lama). Akibatnya `import pyiqa`
gagal sebelum satu metrik pun dihitung:

    AttributeError: `np.sctypes` was removed in the NumPy 2.0 release.

Impor modul ini SEBELUM `import pyiqa`. Bila NumPy < 2 dipakai, modul ini tidak
melakukan apa pun.

Nilai-nilai di bawah menyalin definisi NumPy 1.26 apa adanya, jadi imgaug melihat
daftar dtype yang sama seperti sebelumnya — bukan tebakan.
"""

import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64, np.longdouble],
        "complex": [np.complex64, np.complex128, np.clongdouble],
        "others": [bool, object, bytes, str, np.void],
    }

# Alias dtype yang juga dihapus pada NumPy 2.0 dan masih dipakai imgaug/basicsr.
for _name, _alias in (
    ("bool8", np.bool_),
    ("object0", object),
    ("str0", str),
    ("bytes0", bytes),
    ("void0", np.void),
    ("float_", np.float64),
    ("complex_", np.complex128),
    ("unicode_", np.str_),
    ("Inf", np.inf),
    ("NaN", np.nan),
):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)
