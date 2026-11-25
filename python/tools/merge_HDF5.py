"""Merge per-snapshot TNG chunk files into a single h-free combined_NNN.hdf5, and merge the
other downloaded chunked products (--groupcats, --trees, --ics) so their subfiles can be deleted. Every combined_*
file is h-FREE; --add-fields, --verify and --delete-chunks are described by their --help. Usage: merge_HDF5.py -d DIR"""

import fcntl
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


SNAPSHOT_NUMBERS = [0, 4, 17, 33, 50]

H_VALUE = 0.6774

# what the programs read of a snapshot, in the types a merged file stores
DATASETS_INFO = {
    "PartType1": {
        "Coordinates": {"shape": (None, 3), "dtype": 'float32'},
        "ParticleIDs": {"shape": (None,), "dtype": 'uint64'},
        "Velocities": {"shape": (None, 3), "dtype": 'float32'},
    }
}

# The other per-particle datasets of TNG's -Dark snapshots and the power of h in their raw units (TNG data
# specification, PartType1): h-free value = raw * h^power. Only the fallback: a chunk's own 'h_scaling' attribute
# is read first. A dataset with neither is copied raw, with a loud warning.
_H_POWER = {"Potential": 0.0, "SubfindDMDensity": 2.0, "SubfindDensity": 2.0, "SubfindHsml": -1.0,
            "SubfindVelDisp": 0.0}
# TNG's unit attributes of a dataset (cgs value = raw * a^a_scaling * h^h_scaling * to_cgs), kept on a carried one
_UNIT_ATTRS = ("a_scaling", "h_scaling", "length_scaling", "mass_scaling", "velocity_scaling", "to_cgs")

# Groupcat/tree columns whose raw TNG units carry an h (lengths ckpc/h, masses 1e10 Msun/h,
# spin ckpc/h km/s): divided by h at merge. Curated against the TNG data specification for
# the -Dark column set (groupcats and SubLink extended trees share these names).
_H_DIVIDE = {
    # lengths
    "GroupCM", "GroupPos", "Group_R_Crit200", "Group_R_Crit500", "Group_R_Mean200",
    "Group_R_TopHat200", "SubhaloCM", "SubhaloPos", "SubhaloHalfmassRad",
    "SubhaloHalfmassRadType", "SubhaloVmaxRad",
    # masses
    "GroupMass", "GroupMassType", "Group_M_Crit200", "Group_M_Crit500", "Group_M_Mean200",
    "Group_M_TopHat200", "SubhaloMass", "SubhaloMassInHalfRad", "SubhaloMassInHalfRadType",
    "SubhaloMassInMaxRad", "SubhaloMassInMaxRadType", "SubhaloMassInRad",
    "SubhaloMassInRadType", "SubhaloMassType", "Mass", "MassHistory",
    # spin
    "SubhaloSpin",
}
# h-less columns: velocities (km/s), dispersions, counts, lengths-in-particles, indices, IDs.
_H_NONE = {
    "GroupFirstSub", "GroupLen", "GroupLenType", "GroupNsubs", "GroupVel", "SubhaloGrNr",
    "SubhaloIDMostbound", "SubhaloLen", "SubhaloLenType", "SubhaloParent", "SubhaloVel",
    "SubhaloVelDisp", "SubhaloVmax", "SubhaloFlag",
    "DescendantID", "FirstProgenitorID", "FirstSubhaloInFOFGroupID", "LastProgenitorID",
    "MainLeafProgenitorID", "NextProgenitorID", "NextSubhaloInFOFGroupID", "NumParticles",
    "RootDescendantID", "SnapNum", "SubfindID", "SubhaloID", "SubhaloIDRaw", "TreeID",
}


def _sorted_chunks(directory, pattern):
    """Chunk files sorted by their trailing index -- concatenation order defines the global
    particle/row offsets, so it must match the chunk numbering. A name whose index is not a number (a copy
    made by hand beside the chunks) is not a chunk."""
    chunks = [p for p in Path(directory).glob(pattern) if p.stem.split(".")[-1].isdigit()]
    return sorted(chunks, key=lambda p: int(p.stem.split(".")[-1]))


class ChunkError(RuntimeError):
    """A chunk file that cannot be read -- a download stopped half-way, or a file that is not HDF5."""


# what HDF5 says about a file whose bytes stop early, and the size below which a file that does not
# open cannot be a chunk at all (a real chunk's header alone is many KB) -- download_snapshots.sh's
# prune_incomplete uses THE SAME two rules; keep them in step
_INCOMPLETE = ("truncated file", "file signature not found", "bad byte number")
_MIN_CHUNK_BYTES = 4096


def _is_incomplete(path, exc) -> bool:
    why = str(exc).splitlines()[0]
    if any(k in why for k in _INCOMPLETE):
        return True
    try:
        return Path(path).stat().st_size < _MIN_CHUNK_BYTES
    except OSError:
        return False


def _chunk_error(path, exc):
    why = str(exc).splitlines()[0]
    name = Path(path).name
    if _is_incomplete(path, exc):
        return ChunkError(f"Error: couldn't read {name} (an incomplete download): {why}\n"
                          "  No combined file was written. Run the download again: it removes an incomplete "
                          "chunk and fetches it anew.")
    return ChunkError(f"Error: couldn't read {name} (it cannot be read): {why}\n"
                      f"  No combined file was written. Delete {name} (or fix its permissions) and run the "
                      "download again.")


def _open_chunk(path):
    """h5py.File(path, 'r'), or a ChunkError that names the chunk and says what to do."""
    try:
        return h5py.File(path, "r")
    except OSError as e:
        raise _chunk_error(path, e) from e


def _expected_chunks(found, attr):
    """The chunk count the header of the first chunk found promises ('NumFilesPerSnapshot' for snapshots,
    'NumFiles' for group catalogues), or None. Counting files on disk instead would let a download that stopped
    early merge into a 'complete' file with missing particles."""
    if not found:
        return None
    with _open_chunk(found[0]) as f:
        try:                                # a chunk truncated past its superblock fails here, not at open
            v = f["Header"].attrs.get(attr) if "Header" in f else None
        except (OSError, RuntimeError) as e:
            raise _chunk_error(found[0], e) from e
    return int(v) if v is not None else None


def _refetch(name):
    """How the chunks of a snapshot that has its combined file come back (the download skips such a snapshot)."""
    return (f"run scripts/download_snapshots.sh from a shell with {name} moved aside (it skips a snapshot whose "
            "combined file is there, and the launcher's Data tab would merge the snapshot anew), then move it back")


def _report_missing(input_files, n_header, merged=None):
    """The chunks that are not there. 'merged': the combined file's name, in the modes that work on one that
    exists (--add-fields, --verify), where the download has to be told to fetch."""
    missing = [f for f in input_files if not f.exists()]
    print(f"\nProblem: Can't find {len(missing) or 'any'} subfile(s)"
          + (f" (the header promises {n_header})" if n_header else "") + ":")
    for f in missing:
        print(f"  Missing: {f}")
    if merged:
        print(f"  Nothing was changed. To fetch what is missing, {_refetch(merged)}; or check -d / -n.\n")
    else:
        print("  No combined file was written. Run the download again (it fetches what is missing), "
              "or check -d / -n.\n")


def _partial(output_file):
    """The name a merge writes under until its totals verify: a stopped or failed merge never leaves
    a combined_*.hdf5 that the launcher and the run scripts would take for a complete snapshot."""
    partial = Path(output_file).with_name(Path(output_file).name + ".partial")
    partial.unlink(missing_ok=True)             # a merge stopped half-way last time
    return partial


def _flush(path):
    """A file's (or a folder's) bytes on the disk itself, not only handed to the system: macOS' full flush where
    there is one (a plain fsync stops at the drive's own cache there), else fsync."""
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except (AttributeError, OSError):       # not macOS, or a file system without it
            os.fsync(fd)
    finally:
        os.close(fd)


def _put(partial, output_file):
    """The finished .partial under its final name, its bytes on the disk first: a power cut or an unplugged disk
    right after must not leave that name on a file whose end never arrived. False, the .partial removed and what
    was under the name before as it was, when the disk refuses."""
    try:
        _flush(partial)
        os.replace(partial, output_file)
    except OSError as e:
        partial.unlink(missing_ok=True)
        print(f"\nError: could not put {Path(output_file).name} in place: {e}\n  Nothing was changed.")
        return False
    except BaseException:                       # a Ctrl-C during the flush: no .partial is left
        partial.unlink(missing_ok=True)
        raise
    return True


def _delete_chunks(files, verified, what):
    if not verified:
        print(f"  NOT deleting {what} chunks: the merged totals did not verify.")
        return
    for folder in {f.parent for f in files}:    # the combined file's name on the disk before its sources go
        try:
            _flush(folder)
        except OSError:                         # a file system that cannot flush a folder
            pass
    for f in files:
        f.unlink()
    print(f"  Deleted {len(files)} {what} chunk file(s).")


def _snapshot_chunks(snapshot_str, snapshot_dir, num_subfiles):
    """(the chunk files a snapshot's header promises, the chunks found on disk, the header's count or None). The
    count is -n when given, else the header's, and the files on disk only when the header has none."""
    found = _sorted_chunks(snapshot_dir, f"snap_{snapshot_str}.*.hdf5")
    n_header = _expected_chunks(found, "NumFilesPerSnapshot")
    if num_subfiles and n_header is not None and n_header != num_subfiles:
        print(f"  WARNING: -n {num_subfiles} given, but the header says {n_header} chunk(s) per snapshot")
    n_files = num_subfiles or n_header or len(found)
    return [snapshot_dir / f"snap_{snapshot_str}.{i}.hdf5" for i in range(n_files)], found, n_header


def merge_snapshot_files(snapshot_num, num_subfiles, base_dir, h_value, datasets_info,
                         delete_chunks=False):
    snapshot_str = f"{snapshot_num:03d}"

    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"

    print(f"\n{'='*60}")
    print(f"Processing snapshot {snapshot_str}")
    print(f"Directory: {snapshot_dir}")
    try:
        ok = _merge_snapshot(snapshot_str, snapshot_dir, output_file, num_subfiles, h_value, datasets_info,
                             delete_chunks)
    except ChunkError as e:
        print(f"\n{e}")
        ok = False
    print('='*60)
    return ok


def _merge_snapshot(snapshot_str, snapshot_dir, output_file, num_subfiles, h_value, datasets_info,
                    delete_chunks):
    """merge_snapshot_files' body; True when combined_NNN.hdf5 was written and verified (or is there
    already and its chunks are gone: a merge with --delete-chunks run again)."""
    partial = _partial(output_file)         # first: a merge stopped half-way may have left a big one
    input_files, found, n_header = _snapshot_chunks(snapshot_str, snapshot_dir, num_subfiles)
    if not found and output_file.exists():
        print(f"  {output_file.name} already exists and no chunks are left: nothing to merge")
        return True
    print(f"Merging {len(input_files)} subfiles...")

    if not input_files or any(not f.exists() for f in input_files):
        _report_missing(input_files, n_header)
        return False

    with _open_chunk(input_files[0]) as f_first:
        box_size = f_first['Header'].attrs.get('BoxSize')
        box_size_physical = box_size / h_value

    try:
        total_counts, expected, types, short = _write_snapshot(partial, input_files, h_value, datasets_info,
                                                               box_size_physical)
    except ChunkError:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError) as e:    # the disk, not a chunk (full, gone, read-only); h5py's close raises RuntimeError
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        return False
    except BaseException:                   # whatever else, a Ctrl-C too: never leave a .partial behind
        partial.unlink(missing_ok=True)
        raise
    verified = all(int(expected[t]) == int(total_counts[t]) for t in types)
    print(f"\n  Particles merged: {total_counts[total_counts > 0]}"
          + ("" if verified else f"  (MISMATCH vs header total {expected[sorted(types)]})"))
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the chunks do not hold what the header promises. Run the "
              "download again, then the merge.")
        return False
    if short:                               # a column shorter than the particles would be misaligned for good
        partial.unlink(missing_ok=True)
        print("  No combined file was written: not every particle has a value of " + ", ".join(short)
              + " (a chunk without that dataset). Run the download again, then the merge.")
        return False
    if not _put(partial, output_file):
        return False
    print(f"Saved to: {output_file}")
    print(f"  Physical box size: {box_size_physical:.2f} kpc")
    if delete_chunks:
        _delete_chunks(input_files, verified, "snapshot")
    return True


def _h_power(name, attrs):
    """The power of h in a chunk dataset's raw units (h-free = raw * h^power): its own 'h_scaling' attribute, else
    the table of the known fields; None when neither says."""
    if "h_scaling" in attrs:
        return float(np.asarray(attrs["h_scaling"]).ravel()[0])
    return _H_POWER.get(name)


def _extra_fields(input_files, datasets_info, group_name):
    """{name: (shape tail, dtype, h power or None, TNG's unit attributes)} of the per-particle datasets a group of
    the chunks holds beside the ones in datasets_info: the union over the chunks (a chunk lacking one makes a short
    column, which the merge refuses). The power is None, the field copied raw, when nothing says it."""
    fields = {}
    for fname in input_files:
        with _open_chunk(fname) as f_in:
            try:
                src = f_in.get(group_name)
                if src is None:
                    continue
                for name, obj in src.items():
                    if not isinstance(obj, h5py.Dataset) or name in datasets_info[group_name] or not obj.shape:
                        continue
                    if name in fields:
                        if (obj.shape[1:], obj.dtype) != fields[name][:2]:
                            raise ChunkError(f"Error: {group_name}/{name} has another shape or type in "
                                             f"{Path(fname).name} than in an earlier chunk.\n"
                                             "  No combined file was written.")
                        continue
                    power = _h_power(name, obj.attrs)
                    if power and not np.issubdtype(obj.dtype, np.floating):
                        power = None
                    units = {k: obj.attrs[k] for k in _UNIT_ATTRS if k in obj.attrs}
                    fields[name] = (obj.shape[1:], obj.dtype, power, units)
            except ChunkError:
                raise
            except (KeyError, OSError, RuntimeError) as e:     # a chunk truncated past its superblock
                raise _chunk_error(fname, e) from e
    return fields


def _create_extra(group, name, rows, tail, dtype, power, units):
    """A dataset for a carried field at its full size, with TNG's unit attributes: 'h_scaling' 0 and the power
    applied once the h is out; as they are for a field without a power, which is copied raw."""
    d = group.create_dataset(name, shape=(rows,) + tuple(tail), dtype=dtype)
    for k, v in units.items():
        d.attrs[k] = 0.0 if k == "h_scaling" and power is not None else v
    if power is None:
        print(f"  WARNING: {group.name}/{name} cannot be made h-free (no 'h_scaling' attribute and not a known field, "
              "or whole numbers with an h in their unit) -- copied RAW; classify it in merge_HDF5.py (_H_POWER) "
              "before trusting its values")
    else:
        d.attrs["h_power_applied"] = power
    return d


def _converted(data, power, h_value, dtype):
    """A chunk's values of a carried field, h-free: times h^power in double, stored in the source type; as they
    are without a power (0, or None)."""
    if not power:
        return data
    return (data.astype(np.float64) * h_value ** power).astype(dtype)


def _merged_values(name, data, h_value, extras):
    """What a merge writes for a chunk's block of dataset 'name': Coordinates over h, a carried field h-free
    (_converted), every other dataset as it is. --verify compares a combined file with exactly this."""
    if name == "Coordinates":
        return data / h_value if h_value else data
    if name in extras:
        _tail, dtype, power, _units = extras[name]
        return _converted(data, power, h_value, dtype)
    return data


def _chunk_rows(input_files, names_of):
    """{group: {dataset: rows}} summed over the chunks, from the datasets' shapes alone (no particle is read).
    A dataset a chunk lacks adds nothing there, which leaves its column short of the particles'."""
    rows = {g: {n: 0 for n in names} for g, names in names_of.items()}
    for fname in input_files:
        with _open_chunk(fname) as f_in:
            try:
                for g, names in rows.items():
                    if g in f_in:
                        for n in names:
                            if n in f_in[g]:
                                rows[g][n] += f_in[g][n].shape[0]
            except (KeyError, OSError, RuntimeError) as e:     # a chunk truncated past its superblock
                raise _chunk_error(fname, e) from e
    return rows


def _write_snapshot(output_file, input_files, h_value, datasets_info, box_size_physical):
    """The merged file at 'output_file'. Returns (counts merged per type, the header's totals, the particle types
    merged, the datasets not holding a value for every particle). A set not holding what its header promises is
    refused before a particle is written; a chunk that fails to read or changes raises ChunkError."""
    with h5py.File(output_file, 'w') as f_out:
        with _open_chunk(input_files[0]) as f_first:
            for group_name in ["Config", "Parameters"]:
                if group_name in f_first:
                    f_out.copy(f_first[group_name], group_name)

            if "Header" in f_first:
                f_out.copy(f_first["Header"], "Header")

        f_out["Header"].attrs["BoxSize"] = box_size_physical
        # MassTable is 1e10 Msun/h in the raw chunks: convert like everything else and mark
        # the file, so FieldSet can tell h-free headers from pre-conversion combined files
        f_out["Header"].attrs["MassTable"] = np.asarray(f_out["Header"].attrs["MassTable"]) / h_value
        f_out["Header"].attrs["HFreeUnits"] = h_value

        # the other per-particle datasets of the chunks: carried too, h-free (see the top)
        extras = {g: _extra_fields(input_files, datasets_info, g) for g in datasets_info}
        rows = _chunk_rows(input_files, {g: list(datasets_info[g]) + list(extras[g]) for g in datasets_info})
        total_counts = np.zeros(6, dtype=np.int64)
        for group_name in datasets_info:
            total_counts[int(group_name[-1])] = rows[group_name]["ParticleIDs"]

        header = f_out["Header"]
        # chunk-0 global counts; TNG splits them into a low and a high 32-bit word (the -1 runs exceed 2^32)
        expected = np.array(header.attrs["NumPart_Total"], dtype=np.int64)
        if "NumPart_Total_HighWord" in header.attrs:
            expected += np.array(header.attrs["NumPart_Total_HighWord"], dtype=np.int64) << 32
            header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)   # the totals below are int64
        header.attrs["NumPart_ThisFile"] = total_counts
        header.attrs["NumPart_Total"] = total_counts
        types = {int(g[-1]) for g in datasets_info}       # a merged type with 0 particles must verify too
        # every dataset of a merged type holds one row a particle, or the merge is refused
        short = [f"{g}/{name}" for g, names in rows.items() for name, r in names.items()
                 if r != total_counts[int(g[-1])]]
        if short or any(int(expected[t]) != int(total_counts[t]) for t in types):
            return total_counts, expected, types, short

        out_datasets = {}
        for group_name, dsets in datasets_info.items():
            group = f_out.create_group(group_name)
            n = int(total_counts[int(group_name[-1])])
            out_datasets[group_name] = {
                name: group.create_dataset(name, shape=(n,) + tuple(props["shape"][1:]), dtype=props["dtype"])
                for name, props in dsets.items()}
            for name, (tail, dtype, power, units) in extras[group_name].items():
                out_datasets[group_name][name] = _create_extra(group, name, n, tail, dtype, power, units)
            if extras[group_name]:
                print(f"  {group_name}: also carrying {', '.join(sorted(extras[group_name]))}")

        filled = {g: {name: 0 for name in dsets} for g, dsets in out_datasets.items()}
        for file_idx, fname in enumerate(input_files):
            progress = f"[{file_idx + 1}/{len(input_files)}]"
            print(f"  {progress} Reading {fname.name}...")

            with _open_chunk(fname) as f_in:
                for group_name, dsets in out_datasets.items():
                    for name, dset in dsets.items():
                        try:                        # a chunk truncated past its superblock fails here, not at open
                            have = group_name in f_in and name in f_in[group_name]
                            data = f_in[group_name][name][...] if have else None
                            n = f_in[group_name]["ParticleIDs"].shape[0] if have else 0
                        except (KeyError, OSError, RuntimeError) as e:
                            raise _chunk_error(fname, e) from e
                        if not have:
                            continue                # a chunk without particles of this type
                        if data.shape[0] != n:      # rows that shift between chunks: the totals may still add up
                            raise ChunkError(f"Error: {fname.name} holds {data.shape[0]} rows of {group_name}/{name}"
                                             f", its particles {n}.\n  No combined file was written.")
                        data = _merged_values(name, data, h_value, extras[group_name])
                        at = filled[group_name][name]
                        if at + data.shape[0] > dset.shape[0]:      # the chunk changed since its rows were counted
                            raise ChunkError(f"Error: {fname.name} holds more rows of {group_name}/{name} than "
                                             "when the merge began.\n  No combined file was written.")
                        dset[at:at + data.shape[0], ...] = data
                        filled[group_name][name] = at + data.shape[0]
        unfilled = [f"{g}/{name}" for g, dsets in out_datasets.items() for name, dset in dsets.items()
                    if filled[g][name] != dset.shape[0]]
        if unfilled:                                                # ... or lost rows since
            raise ChunkError(f"Error: the chunks hold fewer rows of {', '.join(unfilled)} than when the merge "
                             "began.\n  No combined file was written.")
    return total_counts, expected, types, short


def _chunk_problem(e, merged):
    """A ChunkError's own line, with what to do in the modes that work on a combined file that exists."""
    print(f"\n{str(e).splitlines()[0]}\n  Nothing was changed. To fetch a damaged chunk anew, delete it and "
          f"{_refetch(merged)}.")


def add_snapshot_fields(snapshot_num, num_subfiles, base_dir, datasets_info):
    """--add-fields for one snapshot: the per-particle datasets its chunks hold beside the merged ones, put into the
    combined file that is there already. True when the file carries them all afterwards; None when no chunk is
    left and the file carries none. No chunk is deleted here: that is --verify's."""
    snapshot_str = f"{snapshot_num:03d}"
    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"
    print(f"\n{'='*60}")
    print(f"Adding the other particle fields to snapshot {snapshot_str}")
    print(f"Directory: {snapshot_dir}")
    partial = _partial(output_file)
    ok = False
    try:
        ok = _add_fields(snapshot_str, snapshot_dir, output_file, partial, num_subfiles, datasets_info)
    except ChunkError as e:
        _chunk_problem(e, output_file.name)
    except (KeyError, OSError, RuntimeError) as e:      # the combined file or the disk, not a chunk
        print(f"\nError: could not read {output_file.name} or write its copy {partial.name}: {e}\n"
              "  Nothing was changed.")
    finally:                                            # whatever happened, a Ctrl-C too: no .partial is left
        partial.unlink(missing_ok=True)
    print('='*60)
    return ok


_BLOCK = 1 << 22        # rows a pass over a dataset reads at a time


def _same_datasets(f_a, f_b):
    """Whether every dataset of f_a is in f_b with the same shape, type and values, read in blocks; the first
    difference's name, else ''. The proof that a rewritten file kept what it had."""
    names = []
    f_a.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    for name in names:
        a, b = f_a[name], f_b.get(name)
        if b is None or a.shape != b.shape or a.dtype != b.dtype:
            return name
        if a.shape == ():
            if not np.array_equal(a[()], b[()]):
                return name
            continue
        for i in range(0, a.shape[0], _BLOCK):
            if a[i:i + _BLOCK].tobytes() != b[i:i + _BLOCK].tobytes():      # bit for bit (a NaN equals itself here)
                return name
    return ""


def _copy_laid_out(src, dst_group, name):
    """Dataset 'src' into dst_group in one piece (not in the small pieces an older merge grew it by), its type,
    values and attributes as they are."""
    d = dst_group.create_dataset(name, shape=src.shape, dtype=src.dtype)
    for k, v in src.attrs.items():
        d.attrs[k] = v
    if src.shape == ():
        d[()] = src[()]
    for i in range(0, src.shape[0] if src.shape else 0, _BLOCK):
        d[i:i + _BLOCK] = src[i:i + _BLOCK]
    return d


def _add_fields(snapshot_str, snapshot_dir, output_file, partial, num_subfiles, datasets_info):
    """add_snapshot_fields' body. A copy grows beside the combined file ('partial', removed by the caller) and
    replaces it only when particles match ID for ID, added datasets have one row a particle and old datasets are bit
    for bit in the copy; it keeps the file's time stamp, so no grid built from it looks out of date."""
    if not output_file.exists():
        print(f"  Problem: there is no {output_file.name} to add to. Merge the snapshot first (without --add-fields).")
        return False
    input_files, found, n_header = _snapshot_chunks(snapshot_str, snapshot_dir, num_subfiles)
    stamp = output_file.stat()
    with h5py.File(output_file, "r") as f_old:
        if not found:
            have = [n for g in datasets_info if g in f_old for n in sorted(set(f_old[g]) - set(datasets_info[g]))]
            if have:
                print(f"  {output_file.name} carries {', '.join(have)} and no chunks are left: nothing to add")
                return True
            print(f"  No chunk of snapshot {snapshot_str} is left, and {output_file.name} carries nothing beside "
                  "positions, velocities and IDs: nothing to add from.\n  If its chunks held other fields (TNG's "
                  f"full snapshots do), {_refetch(output_file.name)}; then --add-fields.")
            return None
        if any(not f.exists() for f in input_files):
            _report_missing(input_files, n_header, output_file.name)
            return False
        h_value = float(f_old["Header"].attrs.get("HFreeUnits", 0.0))
        if not h_value:
            print(f"  Problem: {output_file.name} has no 'HFreeUnits' marker (a merge from before the unit "
                  "conversion). Merge the snapshot anew instead.")
            return False
        todo = {}
        for g in datasets_info:
            fields = _extra_fields(input_files, datasets_info, g)
            todo[g] = {n: v for n, v in fields.items() if g not in f_old or n not in f_old[g]}
        if not any(todo.values()):
            print(f"  {output_file.name} already carries every per-particle dataset of its chunks: nothing to add")
            return True
        print("  Adding " + ", ".join(f"{g}/{n}" for g, v in todo.items() for n in sorted(v)))
        # before anything is written: one row a particle in every dataset to add, room for the copy, and the
        # chunks' particles the file's own, ID for ID in order
        rows = _chunk_rows(input_files, {g: ["ParticleIDs"] + list(v) for g, v in todo.items() if v})
        count = {g: f_old[g]["ParticleIDs"].shape[0] if g in f_old else -1 for g in rows}
        short = sorted(f"{g}/{n}" for g, names in rows.items() for n, r in names.items() if r != count[g])
        if short:
            print("\n  Problem: not every particle has a value of " + ", ".join(short)
                  + " (the chunks hold another number of particles than the file, or one lacks the dataset). "
                  "Nothing was changed.")
            return False
        need = stamp.st_size + sum(count[g] * np.dtype(dtype).itemsize * int(np.prod(tail, dtype=np.int64))
                                   for g in rows for tail, dtype, _power, _units in todo[g].values())
        free = shutil.disk_usage(snapshot_dir).free
        if free < need:
            print(f"\n  Problem: the copy needs {need / 1e9:.1f} GB beside the file, and the disk has "
                  f"{free / 1e9:.1f} GB free. Nothing was changed.")
            return False
        at = {g: 0 for g in rows}
        for fname in input_files:
            with _open_chunk(fname) as f_in:
                for g in rows:
                    if g not in f_in:
                        continue
                    try:
                        ids = f_in[g]["ParticleIDs"][...]
                    except (KeyError, OSError, RuntimeError) as e:
                        raise _chunk_error(fname, e) from e
                    if not np.array_equal(f_old[g]["ParticleIDs"][at[g]:at[g] + ids.shape[0]], ids):
                        print(f"\n  Problem: the particles of {fname.name} are not the ones at row {at[g]} "
                              f"of {output_file.name} (other IDs or another order): these chunks are not "
                              "the ones it was merged from. Nothing was changed.")
                        return False
                    at[g] += ids.shape[0]
        with h5py.File(partial, "w") as f_out:
            for k, v in f_old.attrs.items():
                f_out.attrs[k] = v
            for name in f_old:
                if name not in datasets_info:           # Header, Config, Parameters: as they are
                    f_old.copy(f_old[name], f_out, name)
                    continue
                group = f_out.create_group(name)
                for k, v in f_old[name].attrs.items():
                    group.attrs[k] = v
                for dname, obj in f_old[name].items():
                    if isinstance(obj, h5py.Dataset):
                        _copy_laid_out(obj, group, dname)
                    else:
                        f_old.copy(obj, group, dname)
            new = {g: {n: _create_extra(f_out.require_group(g), n, count[g], *v) for n, v in todo[g].items()}
                   for g in rows}
            at = {g: 0 for g in rows}
            for idx, fname in enumerate(input_files):
                print(f"  [{idx + 1}/{len(input_files)}] Reading {fname.name}...")
                with _open_chunk(fname) as f_in:
                    for g in rows:
                        if g not in f_in:
                            continue
                        for name, dset in new[g].items():
                            try:
                                n = f_in[g]["ParticleIDs"].shape[0]
                                data = f_in[g][name][...]
                            except (KeyError, OSError, RuntimeError) as e:
                                raise _chunk_error(fname, e) from e
                            if data.shape[0] != n:
                                raise ChunkError(f"Error: {fname.name} holds {data.shape[0]} rows of {g}/{name}, its "
                                                 f"particles {n}.")
                            if at[g] + n > count[g]:                # the chunk changed since the ID pass
                                raise ChunkError(f"Error: {fname.name} holds more particles than when they were "
                                                 "compared.")
                            dset[at[g]:at[g] + n, ...] = _merged_values(name, data, h_value, todo[g])
                        at[g] += n
            lost = [g for g in rows if at[g] != count[g]]
            if lost:
                raise ChunkError(f"Error: the chunks hold fewer particles of {', '.join(lost)} than when they "
                                 "were compared.")
        with h5py.File(partial, "r") as f_new:
            differs = _same_datasets(f_old, f_new)
        if differs:
            print(f"\n  Problem: {differs} did not come through the copy bit for bit. Nothing was changed.")
            return False
    os.utime(partial, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))    # before the name: never a new time on show
    if not _put(partial, output_file):
        return False
    print(f"\n  Added to: {output_file}")
    print("  Positions, velocities and IDs are the file's own, bit for bit, and it keeps its time stamp: the grids "
          "built from it stay current.\n  The file is larger now: the programs' cached tessellations of this "
          "snapshot are built anew once (the old ones can be deleted), and a killed run of it can no longer "
          "--resume.")
    return True


def verify_snapshot(snapshot_num, num_subfiles, base_dir, datasets_info, delete_chunks=False):
    """--verify for one snapshot: does combined_NNN.hdf5 hold everything its chunks do? Read only, but for
    --delete-chunks: the chunks of a snapshot that compares the same are deleted, and of no other. Returns 'same',
    'no chunks' (none is left to compare with), or what did not come out, in the words of the run's summary."""
    snapshot_str = f"{snapshot_num:03d}"
    snapshot_dir = Path(base_dir) / f"snapdir_{snapshot_str}"
    output_file = snapshot_dir / f"combined_{snapshot_str}.hdf5"
    print(f"\n{'='*60}")
    print(f"Comparing snapshot {snapshot_str} with its chunks")
    print(f"Directory: {snapshot_dir}")
    same, chunks = False, []
    try:
        same, chunks = _verify(snapshot_str, snapshot_dir, output_file, num_subfiles, datasets_info)
    except ChunkError as e:
        _chunk_problem(e, output_file.name)
    except (KeyError, OSError, RuntimeError) as e:      # the combined file, not a chunk
        print(f"\n  Problem: {output_file.name} cannot be read: {e}")
    status = "no chunks" if same is None else "same" if same else "not verified against its chunks"
    if delete_chunks and same:
        try:
            _flush(output_file)                         # its bytes on the disk itself before its sources go
            _delete_chunks(chunks, True, "snapshot")
        except OSError as e:
            print(f"  Problem: the chunks of snapshot {snapshot_str} could not be deleted: {e}")
            status = "its chunks could not be deleted"
    elif delete_chunks and same is False:
        print(f"  NOT deleting the chunks of snapshot {snapshot_str}.")
    print('='*60)
    return status


def _verify(snapshot_str, snapshot_dir, output_file, num_subfiles, datasets_info):
    """verify_snapshot's body: (True, False or None; the chunk files). Chunk by chunk, in order: the particle IDs
    the file's and every dataset of the chunk in the file with exactly the values a merge writes for it (bit for
    bit); then no row beyond the chunks', no particle type the file lacks, and the header the chunks'."""
    if not output_file.exists():
        print(f"  Problem: there is no {output_file.name} to compare.")
        return False, []
    input_files, found, n_header = _snapshot_chunks(snapshot_str, snapshot_dir, num_subfiles)
    if not found:
        print(f"  No chunk of snapshot {snapshot_str} is left: nothing to compare {output_file.name} with (and "
              "nothing to delete).")
        return None, []
    if any(not f.exists() for f in input_files):
        _report_missing(input_files, n_header, output_file.name)
        return False, input_files
    with _open_chunk(input_files[0]) as f_first:
        try:                                # a chunk truncated past its superblock fails here, not at open
            first = dict(f_first["Header"].attrs)
        except (KeyError, OSError, RuntimeError) as e:
            raise _chunk_error(input_files[0], e) from e
    problems = []
    with h5py.File(output_file, "r") as f_m:
        head = f_m["Header"].attrs
        h_value = float(head.get("HFreeUnits", 0.0))
        if not h_value:                     # such a file has its positions and box over h and its masses not
            print(f"  Problem: {output_file.name} has no 'HFreeUnits' marker (a merge from before the unit "
                  "conversion): not compared. Merge the snapshot anew.")
            return False, input_files
        for key, per_h in (("BoxSize", True), ("Redshift", False), ("Time", False)):
            if key not in first:
                continue
            want = float(first[key]) / (h_value if per_h else 1.0)
            if key not in head:
                problems.append(f"the header has no {key}")
            elif abs(float(head[key]) - want) > 1e-12 * max(1.0, abs(want)):
                problems.append(f"the header's {key} is {float(head[key]):.10g}, the chunks give {want:.10g}")
        table = np.asarray(first["MassTable"]) / h_value if "MassTable" in first else None
        if table is not None and ("MassTable" not in head
                                  or not np.allclose(np.asarray(head["MassTable"]), table, rtol=1e-12, atol=0.0)):
            problems.append("the header's MassTable is not the chunks' over h")
        # the particles the chunks' own header counts, type by type: one the file does not carry (a full-physics
        # run's gas, stars, ...) is a problem here, the merged types' counts further down
        low = np.asarray(first.get("NumPart_Total", ()), dtype=np.int64).ravel()
        high = np.asarray(first.get("NumPart_Total_HighWord", ()), dtype=np.int64).ravel()
        promised = [int(low[t]) + (int(high[t]) << 32 if t < high.size else 0) for t in range(low.size)]
        for t, n in enumerate(promised):
            if n and f"PartType{t}" not in datasets_info:
                problems.append(f"the chunks hold {n} particles of PartType{t}, which a merged snapshot does not carry")
        extras = {g: _extra_fields(input_files, datasets_info, g) for g in datasets_info}
        at = {g: 0 for g in datasets_info}
        absent = set()                      # the chunks' datasets the file does not carry at all
        for idx, fname in enumerate(input_files):
            bad = []
            with _open_chunk(fname) as f_in:
                for g in datasets_info:
                    if g not in f_in:
                        continue
                    try:
                        n = f_in[g]["ParticleIDs"].shape[0]
                        names = {name: obj.shape for name, obj in f_in[g].items() if isinstance(obj, h5py.Dataset)}
                    except (KeyError, OSError, RuntimeError) as e:
                        raise _chunk_error(fname, e) from e
                    for name, shape in names.items():
                        if not shape:
                            bad.append(f"{g}/{name} is no per-particle dataset: a merged snapshot does not carry it")
                            continue
                        if g not in f_m or name not in f_m[g]:
                            absent.add(f"{g}/{name}")
                            continue
                        merged = f_m[g][name]
                        if shape[0] != n:
                            bad.append(f"{g}/{name} holds {shape[0]} rows, the chunk's particles {n}")
                            continue
                        if at[g] + n > merged.shape[0]:
                            bad.append(f"{g}/{name} has {merged.shape[0]} rows, the chunks hold more")
                            continue
                        for i in range(0, n, _BLOCK):
                            try:
                                data = f_in[g][name][i:i + _BLOCK]
                            except (OSError, RuntimeError) as e:
                                raise _chunk_error(fname, e) from e
                            want = np.asarray(_merged_values(name, data, h_value, extras[g])).astype(merged.dtype)
                            if merged[at[g] + i:at[g] + i + want.shape[0]].tobytes() != want.tobytes():
                                bad.append(f"{g}/{name} differs within rows {at[g] + i} to {at[g] + i + want.shape[0]}")
                                break
                    at[g] += n
            print(f"  [{idx + 1}/{len(input_files)}] {fname.name}: "
                  + ("; ".join(bad) if bad else "the same" + (" (in what the file carries)" if absent else "")))
            problems += [f"{fname.name}: {b}" for b in bad]
        if absent:
            problems.append(f"{output_file.name} does not carry {', '.join(sorted(absent))}")
        for g in datasets_info:
            t = int(g[-1])
            if t < len(promised) and promised[t] != at[g]:
                problems.append(f"the chunks' header counts {promised[t]} particles of type {t}, the chunks hold "
                                f"{at[g]}")
            if g not in f_m:
                continue
            for name, obj in f_m[g].items():
                if isinstance(obj, h5py.Dataset) and obj.shape[:1] != (at[g],):
                    problems.append(f"{g}/{name} has {obj.shape[0] if obj.shape else 'no'} rows, the chunks hold "
                                    f"{at[g]} particles")
            totals = head.get("NumPart_Total")
            total = int(np.asarray(totals).ravel()[t]) if totals is not None else -1
            if total != at[g]:
                problems.append(f"the header counts {total if total >= 0 else 'no'} particles of type {t}, the "
                                f"chunks hold {at[g]}")
        carried = {g: sorted(f_m[g]) for g in datasets_info if g in f_m}
    if problems:
        print(f"\n  NOT the same: {output_file.name} does not hold everything its {len(input_files)} chunks do.")
        for p in problems[:12]:
            print(f"    {p}")
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more")
        if absent:
            print("  --add-fields puts them into the file.")
    else:
        print(f"\n  VERIFIED: {output_file.name} holds everything its {len(input_files)} chunks do, in their order ("
              + "; ".join(f"{g}: {at[g]} particles, {', '.join(v)}" for g, v in carried.items())
              + "); positions in the file's precision.")
    return not problems, input_files


def _concat_datasets(f_out, out_group, columns, chunk_files, h_value, group_path=None):
    """Concatenate `columns` ({name: (shape_tail, dtype)}) across `chunk_files` in order, dividing the _H_DIVIDE
    columns by h (float64, stored in the source dtype; marker attribute 'divided_by_h' = h).
    Returns {name: rows_written}."""
    dsets, written, convert = {}, {}, {}
    for name, (tail, dtype) in columns.items():
        dsets[name] = out_group.create_dataset(name, shape=(0,) + tail,
                                               maxshape=(None,) + tail,
                                               dtype=dtype, chunks=True)
        written[name] = 0
        convert[name] = name in _H_DIVIDE
        if convert[name] and not np.issubdtype(np.dtype(dtype), np.floating):
            print(f"  WARNING: {name} is marked for h-conversion but stored as {dtype}; copied raw")
            convert[name] = False
        elif convert[name]:
            dsets[name].attrs["divided_by_h"] = h_value
        elif name not in _H_NONE:
            print(f"  WARNING: column {name} is in neither unit map -- copied RAW (ckpc/h?); "
                  f"classify it in merge_HDF5.py before trusting its values")
    for idx, path in enumerate(chunk_files):
        print(f"  [{idx + 1}/{len(chunk_files)}] {path.name}")
        with _open_chunk(path) as f_in:
            src = f_in.get(group_path) if group_path else f_in
            if src is None:
                continue        # a chunk with zero groups/subhalos has no such group at all
            for name, d in dsets.items():
                if name not in src:
                    continue    # a chunk with zero rows may omit the dataset entirely
                try:
                    data = src[name][...]
                except OSError as e:        # truncated past the superblock
                    raise _chunk_error(path, e) from e
                if convert[name]:
                    data = (data.astype(np.float64) / h_value).astype(d.dtype)
                d.resize((written[name] + data.shape[0],) + d.shape[1:])
                d[written[name]:, ...] = data
                written[name] += data.shape[0]
    return written


def _collect_columns(chunk_files, group_path=None):
    """{name: (shape_tail, dtype)} union across chunks (zero-row chunks omit datasets)."""
    columns = {}
    for path in chunk_files:
        with _open_chunk(path) as f:
            src = f.get(group_path) if group_path else f
            if src is None:
                continue
            for name, obj in src.items():
                if isinstance(obj, h5py.Dataset) and name not in columns:
                    columns[name] = (obj.shape[1:], obj.dtype)
        if columns:
            break   # the first chunk that has the group defines it; later ones only append
    return columns


def merge_groupcat_files(snapshot_num, base_dir, delete_chunks=False):
    """groups_NNN/fof_subhalo_tab_NNN.X.hdf5 -> combined_fof_subhalo_tab_NNN.hdf5, h-FREE
    (lengths/masses/spins and the header BoxSize divided by h, marker attributes set);
    dtfelib.groupcat restores its documented raw-ckpc/h API from the markers when reading."""
    snapshot_str = f"{snapshot_num:03d}"
    group_dir = Path(base_dir) / f"groups_{snapshot_str}"
    output_file = group_dir / f"combined_fof_subhalo_tab_{snapshot_str}.hdf5"
    print(f"\n{'='*60}")
    try:
        ok = _merge_groupcat(snapshot_str, group_dir, output_file, delete_chunks)
    except ChunkError as e:
        print(f"\n{e}")
        ok = False
    print('='*60)
    return ok


def _merge_groupcat(snapshot_str, group_dir, output_file, delete_chunks):
    partial = _partial(output_file)
    found = _sorted_chunks(group_dir, f"fof_subhalo_tab_{snapshot_str}.*.hdf5")
    if not found and output_file.exists():
        print(f"  {output_file.name} already exists and no chunks are left: nothing to merge")
        return True
    n_header = _expected_chunks(found, "NumFiles")
    chunk_files = ([group_dir / f"fof_subhalo_tab_{snapshot_str}.{i}.hdf5" for i in range(n_header)]
                   if n_header else found)
    print(f"Merging group catalog {snapshot_str}: {len(chunk_files)} chunk(s)")
    if not chunk_files or any(not f.exists() for f in chunk_files):
        _report_missing(chunk_files, n_header)
        return False

    try:
        rows, n_g, n_s, expected_g, expected_s = _write_groupcat(partial, chunk_files)
    except ChunkError:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError) as e:    # the disk, not a chunk
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        return False
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    verified = (n_g == expected_g and n_s == expected_s
                and all(v in (0, n_g) for v in rows["Group"].values())
                and all(v in (0, n_s) for v in rows["Subhalo"].values()))
    print(f"  Groups: {n_g}/{expected_g}  Subhalos: {n_s}/{expected_s}  "
          f"({'verified' if verified else 'MISMATCH'})")
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the chunks do not hold what the header promises.")
        return False
    if not _put(partial, output_file):
        return False
    print(f"  Saved: {output_file}")
    if delete_chunks:
        _delete_chunks(chunk_files, verified, "group catalog")
    return True


def _write_groupcat(output_file, chunk_files):
    with h5py.File(output_file, "w") as f_out:
        with _open_chunk(chunk_files[0]) as f_first:
            for name in f_first:
                if name not in ("Group", "Subhalo"):
                    f_out.copy(f_first[name], name)     # Header, Config, Parameters, IDs, ...
            expected_g = int(f_first["Header"].attrs["Ngroups_Total"])
            expected_s = int(f_first["Header"].attrs["Nsubgroups_Total"])
            h_value = float(f_first["Header"].attrs["HubbleParam"])
        rows = {}
        for grp in ("Group", "Subhalo"):
            columns = _collect_columns(chunk_files, grp)
            rows[grp] = _concat_datasets(f_out, f_out.create_group(grp), columns,
                                         chunk_files, h_value, grp) if columns else {}
        n_g = max(rows["Group"].values(), default=0)
        n_s = max(rows["Subhalo"].values(), default=0)
        f_out["Header"].attrs["Ngroups_ThisFile"] = n_g
        f_out["Header"].attrs["Nsubgroups_ThisFile"] = n_s
        f_out["Header"].attrs["NumFiles"] = 1
        f_out["Header"].attrs["BoxSize"] = f_out["Header"].attrs["BoxSize"] / h_value
        f_out["Header"].attrs["HFreeUnits"] = h_value
    return rows, n_g, n_s, expected_g, expected_s


def _find_h(base_dir, fallback):
    """HubbleParam from any header on disk (tree chunks carry no cosmology header)."""
    base = Path(base_dir)
    candidates = (list(base.glob("groups_*/fof_subhalo_tab_*.hdf5"))
                  + list(base.glob("snapdir_*/combined_*.hdf5"))
                  + list(base.glob("snapdir_*/snap_*.hdf5"))
                  + list(base.glob("combined_ics.hdf5")) + list(base.glob("ics.hdf5")))
    for path in candidates:
        try:
            with h5py.File(path, "r") as f:
                return float(f["Header"].attrs["HubbleParam"])
        except (OSError, KeyError):
            continue
    print(f"  WARNING: no header with HubbleParam under {base_dir}; using h = {fallback}")
    return fallback


def merge_tree_files(base_dir, delete_chunks=False, h_fallback=H_VALUE):
    """tree_extended.X.hdf5 -> combined_tree_extended.hdf5 (same directory), h-FREE (columns divided by h with
    marker attributes). Each SubLink tree is wholly contained in one chunk and stored contiguously, so
    concatenating the chunks in index order preserves the within-tree row arithmetic dtfelib.trees relies on."""
    base = Path(base_dir)
    tree_dir = None
    cands = (base / "Merger Trees", base / "postprocessing" / "trees" / "SubLink")
    for cand in cands:
        if cand.is_dir() and any(cand.glob("tree_extended.*.hdf5")):
            tree_dir = cand
            break
    print(f"\n{'='*60}")
    if tree_dir is None:
        done = next((c / "combined_tree_extended.hdf5" for c in cands if (c / "combined_tree_extended.hdf5").exists()), None)
        if done is not None:            # merged with --delete-chunks earlier: nothing left to do
            print(f"Merging trees: {done} already exists and no chunks are left: nothing to merge")
            print('='*60)
            return True
        print(f"Merging trees: no tree_extended.*.hdf5 under {base} (download_snapshots.sh -t), nothing merged")
        print('='*60)
        return False
    chunk_files = _sorted_chunks(tree_dir, "tree_extended.*.hdf5")
    output_file = tree_dir / "combined_tree_extended.hdf5"
    h_value = _find_h(base_dir, h_fallback)
    print(f"Merging SubLink trees: {len(chunk_files)} chunk(s) in {tree_dir} (h = {h_value})")
    # the tree chunks carry no count of themselves: an unreadable one is caught, a missing one is not
    partial = _partial(output_file)
    try:
        expected = 0
        for path in chunk_files:
            with _open_chunk(path) as f:
                expected += f["SubhaloID"].shape[0]
        with h5py.File(partial, "w") as f_out:
            with _open_chunk(chunk_files[0]) as f_first:
                for name, obj in f_first.items():
                    if not isinstance(obj, h5py.Dataset):
                        f_out.copy(obj, name)               # e.g. a Header group, if present
            f_out.attrs["HFreeUnits"] = h_value
            columns = _collect_columns(chunk_files)
            written = _concat_datasets(f_out, f_out, columns, chunk_files, h_value)
    except ChunkError as e:
        partial.unlink(missing_ok=True)
        print(f"\n{e}")
        print('='*60)
        return False
    except (OSError, RuntimeError) as e:    # the disk, not a chunk
        partial.unlink(missing_ok=True)
        print(f"\nError: could not write {partial}: {e}\n  No combined file was written.")
        print('='*60)
        return False
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    n = written.get("SubhaloID", 0)
    verified = n == expected and all(v == n for v in written.values())
    print(f"  Tree rows: {n}/{expected} across {len(written)} columns "
          f"({'verified' if verified else 'MISMATCH'})")
    if not verified:
        partial.unlink(missing_ok=True)
        print("  No combined file was written: the columns disagree on their row counts.")
        print('='*60)
        return False
    if not _put(partial, output_file):
        print('='*60)
        return False
    print(f"  Saved: {output_file}")
    if delete_chunks:
        _delete_chunks(chunk_files, verified, "tree")
    print('='*60)
    return True


def convert_ics(base_dir, delete_chunks=False):
    """ics.hdf5 / snap_ics.hdf5 -> combined_ics.hdf5 via convert_ic_units.py (h-free, the
    same convention as combined_NNN.hdf5 -- required for PS-DTFE ID matching)."""
    base = Path(base_dir)
    output_file = base / "combined_ics.hdf5"
    print(f"\n{'='*60}")
    if output_file.exists():
        print(f"ICs: {output_file} already exists, skipping")
        print('='*60)
        return True
    raw = next((base / n for n in ("ics.hdf5", "snap_ics.hdf5") if (base / n).exists()), None)
    if raw is None:
        print(f"ICs: no ics.hdf5 / snap_ics.hdf5 in {base} (download_snapshots.sh -i), nothing converted")
        print('='*60)
        return False
    print(f"Converting ICs: {raw} -> {output_file}")
    partial = _partial(output_file)         # the converter opens 'w-': a stale one must be gone
    try:
        rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "convert_ic_units.py"),
                             str(raw), str(partial)]).returncode
    except BaseException:                   # a Ctrl-C: the converter's half-written file goes
        partial.unlink(missing_ok=True)
        raise
    # the final name only once the whole conversion succeeded
    verified = rc == 0 and partial.is_file() and _put(partial, output_file)
    partial.unlink(missing_ok=True)
    if delete_chunks:
        _delete_chunks([raw], verified, "IC")
    if not verified:
        print("  ICs conversion FAILED: no combined_ics.hdf5 was written")
    print('='*60)
    return verified


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # python/: dtfelib
    from dtfelib.cli import new_parser, add_list_arg
    ap = new_parser(description=__doc__.splitlines()[0])
    add_list_arg(ap, "--snaps", kind=int, default=None, metavar="N",
                 help=f"snapshot numbers to merge, space-separated (default: {' '.join(map(str, SNAPSHOT_NUMBERS))})")
    ap.add_argument("-d", "--sim-dir", dest="data_dir", required=True, metavar="DIR",
                    help="the simulation's directory holding snapdir_NNN/, e.g. '~/IllustrisTNG/TNG100/TNG100-3-Dark' "
                         "(required: there is no default simulation; not the data root, the folder of simulations)")
    ap.add_argument("-n", "--num-subfiles", type=int, default=None,
                    help="chunk files per snapshot (default: auto-detected by globbing)")
    ap.add_argument("--hubble", type=float, default=H_VALUE,
                    help=f"fallback h if the file header lacks HubbleParam (default: {H_VALUE})")
    ap.add_argument("--snapshots-too", action="store_true",
                    help="with --groupcats/--trees/--ics: also merge the raw snapshots")
    ap.add_argument("--groupcats", action="store_true",
                    help="merge the groups_NNN chunk files of the given snapshots")
    ap.add_argument("--trees", action="store_true",
                    help="merge the SubLink tree_extended chunks (whole simulation)")
    ap.add_argument("--ics", action="store_true",
                    help="convert ics.hdf5 -> combined_ics.hdf5 (whole simulation)")
    ap.add_argument("--add-fields", action="store_true",
                    help="snapshots: instead of merging, add the per-particle datasets the chunks hold beside "
                         "positions, velocities and IDs to the combined files that are there already (those "
                         "three and the file's time stamp stay as they are; the chunks must still be there, and "
                         "the disk needs room for a copy of the file)")
    ap.add_argument("--verify", action="store_true",
                    help="snapshots: instead of merging, compare each combined file with its chunks, particle for "
                         "particle: read only; exit 1 when one does not hold everything its chunks do (after "
                         "--add-fields when both are given). A snapshot whose chunks are gone has nothing to "
                         "compare and is named")
    ap.add_argument("--delete-chunks", action="store_true",
                    help="delete the source chunk files after a VERIFIED merge (with --add-fields or --verify: "
                         "after the comparison, of the snapshots that compare the same and of no other)")
    args = ap.parse_args()
    snapshots = args.snaps if args.snaps else SNAPSHOT_NUMBERS
    other_modes = args.groupcats or args.trees or args.ics
    if (args.add_fields or args.verify) and other_modes:
        ap.error("--add-fields and --verify work on snapshots alone: drop --groupcats / --trees / --ics")
    do_snapshots = (not other_modes) or args.snapshots_too

    print("\nHDF5 Snapshot Merger")
    print(f"Base directory: {args.data_dir}")
    if do_snapshots or args.groupcats:
        print(f"Snapshots to process: {snapshots}")

    failed = []                 # what asked for a merge and got no verified combined file
    not_done = []               # what --add-fields or --verify was asked for and did not come out
    uncompared = []             # snapshots --verify had no chunk left to compare with
    no_source = []              # ... and --add-fields none to add from, the file carrying no other field
    if args.add_fields or args.verify:
        for snapshot_num in snapshots:
            where = (snapshot_num, args.num_subfiles, args.data_dir, DATASETS_INFO)
            added = add_snapshot_fields(*where) if args.add_fields else True
            if added is None:
                no_source.append(f"{snapshot_num:03d}")
            elif not added:
                not_done.append(f"not added: the other particle fields of snapshot {snapshot_num:03d}")
            elif args.verify or args.delete_chunks:     # a chunk is deleted only after the comparison
                status = verify_snapshot(*where, delete_chunks=args.delete_chunks)
                if status == "no chunks":
                    uncompared.append(f"{snapshot_num:03d}")
                elif status != "same":
                    not_done.append(f"{status}: snapshot {snapshot_num:03d}")
    elif do_snapshots:
        for snapshot_num in snapshots:
            # h from the data itself when available; the CLI value is only a fallback
            first = Path(args.data_dir) / f"snapdir_{snapshot_num:03d}" / f"snap_{snapshot_num:03d}.0.hdf5"
            h_value = args.hubble
            if first.exists():
                try:
                    with _open_chunk(first) as f:
                        h_value = float(f["Header"].attrs.get("HubbleParam", args.hubble))
                except ChunkError as e:
                    print(f"\n{'=' * 60}\nProcessing snapshot {snapshot_num:03d}\n{e}\n{'=' * 60}")
                    failed.append(f"snapshot {snapshot_num:03d}")
                    continue
            if not merge_snapshot_files(
                snapshot_num,
                args.num_subfiles,
                args.data_dir,
                h_value,
                DATASETS_INFO,
                delete_chunks=args.delete_chunks
            ):
                failed.append(f"snapshot {snapshot_num:03d}")

    if args.groupcats:
        for snapshot_num in snapshots:
            if not merge_groupcat_files(snapshot_num, args.data_dir, delete_chunks=args.delete_chunks):
                failed.append(f"group catalogue {snapshot_num:03d}")
    if args.trees:
        if not merge_tree_files(args.data_dir, delete_chunks=args.delete_chunks, h_fallback=args.hubble):
            failed.append("merger trees")
    if args.ics:
        if not convert_ics(args.data_dir, delete_chunks=args.delete_chunks):
            failed.append("initial conditions")

    if no_source:
        print(f"\nNothing to add from, no chunks left: snapshot {', '.join(no_source)}.")
    if uncompared:
        print(f"\nNot compared, no chunks left: snapshot {', '.join(uncompared)}.")
    if failed or not_done:      # the launcher and shell chains see a failed step, not a quiet 'Done!'
        print("\nDone, with problems: " + "; ".join(([f"no combined file for {', '.join(failed)}"] if failed else [])
                                                    + not_done) + ".")
        sys.exit(1)
    print("\nDone!")


if __name__ == "__main__":
    main()
