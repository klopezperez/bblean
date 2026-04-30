# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this software useful please cite the following articles:
# - BitBIRCH: efficient clustering of large molecular libraries:
#   https://doi.org/10.1039/D5DD00030K
# - BitBIRCH Clustering Refinement Strategies:
#   https://doi.org/10.1021/acs.jcim.5c00627
# - BitBIRCh-Lean:
#   (preprint) https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1
#
# Copyright (C) 2025  The Miranda-Quintana Lab and other BitBirch developers, comprised
# exclusively by:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Krisztina Zsigmond <kzsigmond@ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Miroslav Lzicar <miroslav.lzicar@deepmedchem.com>
#
# Authors of this file are:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# version 3 (SPDX-License-Identifier: GPL-3.0-only).
#
# Portions of ./bblean/bitbirch.py are licensed under the BSD 3-Clause License
# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# (SPDX-License-Identifier: BSD-3-Clause). Copies or reproductions of code in the
# ./bblean/bitbirch.py file must in addition adhere to the BSD-3-Clause license terms. A
# copy of the BSD-3-Clause license can be located at the root of this repository, under
# ./LICENSES/BSD-3-Clause.txt.
#
# Portions of ./bblean/bitbirch.py were previously licensed under the LGPL 3.0
# license (SPDX-License-Identifier: LGPL-3.0-only), they are relicensed in this program
# as GPL-3.0, with permission of all original copyright holders:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Vicky (Vic) Jung <jungvicky@ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Kate Huddleston <kdavis2@chem.ufl.edu>
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. This copy can be located at the root of this repository, under
# ./LICENSES/GPL-3.0-only.txt.  If not, see <http://www.gnu.org/licenses/gpl-3.0.html>.
r"""Multi-round BitBirch workflow for clustering huge datasets in parallel"""
import sys
import math
import pickle
import typing as tp
import multiprocessing as mp
from pathlib import Path

from rich.console import Console  # type: ignore
import numpy as np
from numpy.typing import NDArray


from bblean._console import get_console
from bblean._timer import Timer
from bblean._config import DEFAULTS
from bblean.utils import batched
from bblean.bitbirch import BitBirch
from bblean.fingerprints import _get_fps_file_num

__all__ = ["run_multiround_centroids"]


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def _numpy_streaming_save(
    fp_list: list[NDArray[np.integer]] | NDArray[np.integer], path: Path | str
) -> None:
    first_arr = np.ascontiguousarray(fp_list[0])
    header = np.lib.format.header_data_from_array_1_0(first_arr)
    header["shape"] = (len(fp_list), len(first_arr))
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npy")
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        for arr in fp_list:
            np.ascontiguousarray(arr).tofile(f)


# Glob and sort by uint bits and label, if a console is passed then the number of output
# files is printed
def _get_prev_round_centroids_and_n(
    path: Path, round_idx: int, console: Console | None = None
) -> list[tuple[Path, Path]]:
    path = Path(path)
    # TODO: Important: What should be the logic for batching? currently there doesn't
    # seem to be much logic for grouping the files
    centroids_files = sorted(path.glob(f"round-{round_idx - 1}-centroids*.npy"))
    n_files = sorted(path.glob(f"round-{round_idx - 1}-n-*.npy"))
    if console is not None:
        console.print(f"    - Collected {len(centroids_files)} centroid-n file pairs")
    return list(zip(centroids_files, n_files))


def _sort_batch(b: tp.Sequence[tuple[Path, Path]]) -> tuple[tuple[Path, Path], ...]:
    return tuple(
        sorted(
            b, key=lambda x: int(x[0].name.split("-")[-1].split(".")[0]), reverse=True
        )
    )


def _chunk_file_pairs_in_batches(
    file_pairs: tp.Sequence[tuple[Path, Path]],
    bin_size: int,
    console: Console | None = None,
) -> list[tuple[str, tuple[tuple[Path, Path], ...]]]:
    z = len(str(math.ceil(len(file_pairs) / bin_size)))
    # Within each batch, sort the files by starting with the uint16 files, followed by
    # uint8 files, this helps that (approximately) the largest clusters are fitted first
    # which may improve final cluster quality
    batches = [
        (str(i).zfill(z), _sort_batch(b))
        for i, b in enumerate(batched(file_pairs, bin_size))
    ]
    if console is not None:
        console.print(f"    - Chunked files into {len(batches)} batches")
    return batches


def _save_centroids_and_mol_idxs(
    out_dir: Path,
    centroids: list[NDArray[np.integer]],
    mol_ids: list[int],
    label: str,
    round_idx: int,
) -> None:
    _numpy_streaming_save(
        centroids, out_dir / f"round-{round_idx}-centroids-{label}.npy"
    )
    np.save(
        out_dir / f"round-{round_idx}-n-{label}.npy",
        np.array([len(cluster) for cluster in mol_ids]),
    )
    with open(out_dir / f"round-{round_idx}-idxs-{label}.pkl", mode="wb") as f:
        pickle.dump(mol_ids, f)


def _reweight_centroids(
    centroids: NDArray[NDArray[np.integer]],
    sizes: NDArray[np.integer],
    mol_ids: list[int],
) -> list[NDArray[np.integer]]:
    """Recompute centroids as a size-weighted average of original centroids.

    centroids: packed centroids array for the ORIGINAL items (shape: [M, bytes])
    sizes: 1D array with weights for each original centroid (length M)
    mol_ids: list of clusters where each cluster is a sequence of indices into
             the original centroids array

    Returns a list of packed centroids (one per cluster in mol_ids).
    """
    from bblean.fingerprints import unpack_fingerprints, pack_fingerprints

    # Ensure arrays
    orig_centroids = np.asarray(centroids)
    sizes = np.asarray(sizes)

    if orig_centroids.ndim == 1:
        # Single centroid case -> make it 2D
        orig_centroids = orig_centroids.reshape(1, -1)

    # Unpack all original centroids to binary arrays (0/1)
    # Let unpack_fingerprints infer n_features if possible
    unpacked = unpack_fingerprints(orig_centroids)

    new_centroids: list[NDArray[np.integer]] = []

    for cluster in mol_ids:
        if len(cluster) == 0:
            # Shouldn't happen, but skip empty clusters
            continue
        idxs = np.asarray(cluster, dtype=int)
        w = sizes[idxs].astype(np.int64)
        total_w = int(w.sum())
        if total_w == 0:
            # Avoid division by zero: fallback to simple majority vote
            summed = unpacked[idxs].sum(axis=0)
            avg = (summed >= (len(idxs) * 0.5)).astype(np.uint8)
        else:
            # Weighted sum across rows
            # Convert to int64 to avoid overflow
            weighted = (unpacked[idxs].astype(np.int64).T * w).T
            summed = weighted.sum(axis=0)
            # Compute weighted average, add 0.5 and floor -> equivalent to round half up
            avg = np.floor(summed / total_w + 0.5).astype(np.uint8)

        # Pack back into bytes
        packed = pack_fingerprints(avg)
        new_centroids.append(packed)

    return new_centroids


class _InitialRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        out_dir: Path | str,
        n_features: int | None = None,
        max_fps: int | None = None,
        merge_criterion: str = DEFAULTS.merge_criterion,
        input_is_packed: bool = True,
        reclustering_iterations: int = 3,
        extra_threshold: float = 0.025,
    ) -> None:
        # Essentials for user definition
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.merge_criterion = merge_criterion

        # Other user-definable params
        self.max_fps = max_fps
        self.out_dir = Path(out_dir)
        self.n_features = n_features
        self.input_is_packed = input_is_packed

        # Reclustering params
        self.reclustering_iterations = reclustering_iterations
        self.extra_threshold = extra_threshold

    def __call__(self, file_info: tuple[str, Path, int, int]) -> None:
        file_label, fp_file, start_idx, end_idx = file_info

        # First fit the fps in each process, in parallel.
        # `reinsert_indices` required to keep track of mol idxs in different processes.
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )

        range_ = range(start_idx, end_idx)
        tree.fit(
            fp_file,
            reinsert_indices=range_,
            n_features=self.n_features,
            input_is_packed=self.input_is_packed,
            max_fps=self.max_fps,
        )

        # Recluster if requested, this can be useful to recuperate singletons
        if self.reclustering_iterations > 0:
            tree.recluster_inplace(
                iterations=self.reclustering_iterations,
                extra_threshold=self.extra_threshold,
            )

        # Delete internal nodes to release memory
        tree.delete_internal_nodes()

        # Extract the centroids and mol_ids
        output = tree.get_centroids_mol_ids()
        centroids = output["centroids"]
        mol_ids = output["mol_ids"]
        del output

        _save_centroids_and_mol_idxs(self.out_dir, centroids, mol_ids, file_label, 1)


class _TreeMergingRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        round_idx: int,
        out_dir: Path | str,
        merge_criterion: str,
        all_fp_paths: tp.Sequence[Path] = (),
        reclustering_iterations: int = 0,
        extra_threshold: float = 0.0,
        reweight_centroids: bool = False,
    ) -> None:
        self.all_fp_paths = list(all_fp_paths)
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.round_idx = round_idx
        self.out_dir = Path(out_dir)
        self.merge_criterion = merge_criterion
        self.reclustering_iterations = reclustering_iterations
        self.extra_threshold = extra_threshold
        self.reweight_centroids = reweight_centroids

    def __call__(self, batch_info: tuple[str, tp.Sequence[tuple[Path, Path]]]) -> None:
        batch_label, batch_path_pairs = batch_info

        # Put all the sizes in one np array
        sizes = []
        # Collect original centroids for reweighting if needed
        original_centroids_list = []

        # Create new tree
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )

        # Rebuild a tree, inserting all BitFeatures from the corresponding batch
        for centroid_path, sizes_path in batch_path_pairs:
            # Fit the centroids
            tree.fit(centroid_path)

            if self.reweight_centroids:
                # Attach the sizes
                sizes.extend(np.load(sizes_path, mmap_mode="r"))
                # Load the original packed centroids
                original_centroids_list.extend(np.load(centroid_path, mmap_mode="r"))

        if self.reweight_centroids:
            # Convert sizes to np array
            sizes = np.array(sizes)

            # Concatenate original centroids into a single array matching `sizes`
            if original_centroids_list:
                try:
                    orig_centroids = np.vstack(original_centroids_list)
                except Exception:
                    # Fallback to concatenation
                    orig_centroids = np.concatenate(original_centroids_list, axis=0)
            else:
                raise ValueError(
                    "Reweighting requested but no original centroids found"
                )

        # Either do a reclustering step or not
        # if self.reclustering_iterations > 0:
        #    tree.recluster_inplace(
        #        iterations=self.reclustering_iterations,
        #        extra_threshold=self.extra_threshold,
        #    )

        # Release memory
        tree.delete_internal_nodes()

        # Get the centroids
        output = tree.get_centroids_mol_ids()
        centroids = output["centroids"]
        mol_ids = output["mol_ids"]
        del output

        # Fix the mol_ids to reciprocate the original indexes
        # Read the mol_ids files
        mol_ids_files = [
            str(f).replace("centroids", "idxs") for f, _ in batch_path_pairs
        ]
        mol_ids_files = [f.replace("npy", "pkl") for f in mol_ids_files]

        all_mol_ids = []
        for mol_ids_file in mol_ids_files:
            with open(mol_ids_file, mode="rb") as f:
                all_mol_ids.extend(pickle.load(f))

        corrected_mol_ids = []
        for cluster in mol_ids:
            corrected_cluster = []
            for centroid_id in cluster:
                # Get the original mol id from the corresponding file
                corrected_cluster.extend(all_mol_ids[centroid_id])
            corrected_mol_ids.append(corrected_cluster)

        if self.reweight_centroids:
            # Do reweight function to take into account sizes
            # Compute new packed centroids based on original centroids and sizes
            try:
                new_centroids = _reweight_centroids(orig_centroids, sizes, mol_ids)
                centroids = new_centroids
            except Exception:
                pass

        _save_centroids_and_mol_idxs(
            self.out_dir, centroids, corrected_mol_ids, batch_label, self.round_idx
        )


class _FinalTreeMergingRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        merge_criterion: str,
        out_dir: Path | str,
        save_tree: bool,
        save_centroids: bool,
        reclustering_iterations: int = 0,
        extra_threshold: float = 0.0,
    ) -> None:
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.merge_criterion = merge_criterion
        self.out_dir = Path(out_dir)
        self.save_tree = save_tree
        self.save_centroids = save_centroids
        self.reclustering_iterations = reclustering_iterations
        self.extra_threshold = extra_threshold

    def __call__(self, batch_info: tuple[str, tp.Sequence[tuple[Path, Path]]]) -> None:
        batch_label, batch_path_pairs = batch_info

        # Put all the sizes in one np array
        sizes = []

        # Create new tree
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )

        # Rebuild a tree, inserting all BitFeatures from the corresponding batch
        for centroid_path, sizes_path in batch_path_pairs:
            # Fit the centroids
            tree.fit(centroid_path)

            # Attach the sizes
            sizes.extend(np.load(sizes_path, mmap_mode="r"))

        # Convert sizes to np array
        sizes = np.array(sizes)

        # Either do a reclustering step or not
        # if self.reclustering_iterations > 0:
        #    tree.recluster_inplace(
        #        iterations=self.reclustering_iterations,
        #        extra_threshold=self.extra_threshold,
        #    )

        # Save clusters and exit
        if self.save_tree:
            # TODO: Find alternative solution
            tree.save(self.out_dir / "bitbirch.pkl")

        # Release memory
        tree.delete_internal_nodes()

        # Get the clusters
        if self.save_centroids:
            output = tree.get_centroids_mol_ids()
            centroids = output["centroids"]
            with open(self.out_dir / "centroids.pkl", mode="wb") as f:
                pickle.dump(centroids, f)
            mol_ids = output["mol_ids"]
            del output
            del centroids
        else:
            mol_ids = tree.get_cluster_mol_ids()

        # Fix the mol_ids to reciprocate the original indexes
        # Read the mol_ids files
        mol_ids_files = [
            str(f).replace("centroids", "idxs") for f, _ in batch_path_pairs
        ]
        mol_ids_files = [f.replace("npy", "pkl") for f in mol_ids_files]

        all_mol_ids = []
        for mol_ids_file in mol_ids_files:
            with open(mol_ids_file, mode="rb") as f:
                all_mol_ids.extend(pickle.load(f))

        corrected_mol_ids = []
        for cluster in mol_ids:
            corrected_cluster = []
            for centroid_id in cluster:
                # Get the original mol id from the corresponding file
                corrected_cluster.extend(all_mol_ids[centroid_id])
            corrected_mol_ids.append(corrected_cluster)

        # Save the corrected clusters
        with open(self.out_dir / "clusters.pkl", mode="wb") as f:
            pickle.dump(corrected_mol_ids, f)


# Create a list of tuples of labels, file paths and start-end idxs
def _get_files_range_tuples(
    files: tp.Sequence[Path],
) -> list[tuple[str, Path, int, int]]:
    running_idx = 0
    files_info = []
    z = len(str(len(files)))
    for i, file in enumerate(files):
        start_idx = running_idx
        end_idx = running_idx + _get_fps_file_num(file)
        files_info.append((str(i).zfill(z), file, start_idx, end_idx))
        running_idx = end_idx
    return files_info


# NOTE: 'full_refinement_before_midsection' indicates if the refinement of the batches
# is fully done after the tree-merging rounds, or if the data is only split before the
# tree-merging rounds
def run_multiround_centroids(
    input_files: tp.Sequence[Path],
    out_dir: Path,
    n_features: int | None = None,
    input_is_packed: bool = True,
    num_initial_processes: int = 10,
    num_midsection_processes: int | None = None,
    merge_criterion: str = DEFAULTS.merge_criterion,
    branching_factor: int = DEFAULTS.branching_factor,
    threshold: float = DEFAULTS.threshold,
    midsection_threshold_change: float = 0.025,
    reclustering_iterations: int = 3,
    extra_threshold: float = 0.025,
    reweight_centroids: bool = False,
    # Advanced
    num_midsection_rounds: int = 1,
    bin_size: int = 5,
    max_tasks_per_process: int = 1,
    mp_context: tp.Any = None,
    save_tree: bool = False,
    save_centroids: bool = True,
    # Debug
    max_fps: int | None = None,
    verbose: bool = False,
    cleanup: bool = True,
) -> Timer:
    r"""Perform (possibly parallel) multi-round BitBirch clustering

    .. warning::

        The functionality provided by this function is stable, but its API
        (the arguments it takes and its return values) may change in the future.
    """
    if mp_context is None:
        mp_context = mp.get_context("forkserver" if sys.platform == "linux" else None)
    # Returns timing and for the different rounds
    # TODO: Also return peak-rss
    console = get_console(silent=not verbose)

    if num_midsection_processes is None:
        num_midsection_processes = num_initial_processes
    else:
        # Sanity check
        if num_midsection_processes > num_initial_processes:
            raise ValueError("Num. midsection procs. must be <= num. initial processes")

    # Common params to all rounds BitBIRCH
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=branching_factor,
        threshold=threshold,
        merge_criterion=merge_criterion,
        out_dir=out_dir,
    )
    timer = Timer()
    timer.init_timing("total")

    # Get starting and ending idxs for each file, and collect them into tuples
    files_range_tuples = _get_files_range_tuples(input_files)  # correct
    num_files = len(input_files)

    # Initial round of clustering
    round_idx = 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Initial) Round {round_idx}: Cluster initial batch of fingerprints")

    initial_fn = _InitialRound(
        branching_factor=common_kwargs["branching_factor"],
        threshold=common_kwargs["threshold"],
        out_dir=common_kwargs["out_dir"],
        n_features=n_features,
        max_fps=max_fps,
        merge_criterion=merge_criterion,
        input_is_packed=input_is_packed,
        reclustering_iterations=reclustering_iterations,
        extra_threshold=extra_threshold,
    )
    num_ps = min(num_initial_processes, num_files)
    console.print(f"    - Processing {num_files} inputs with {num_ps} processes")
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        if num_ps == 1:
            for tup in files_range_tuples:
                initial_fn(tup)
        else:
            with mp_context.Pool(
                processes=num_ps, maxtasksperchild=max_tasks_per_process
            ) as pool:
                pool.map(initial_fn, files_range_tuples)

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(out_dir)

    ###########################################################

    # Mid-section "Tree-Merging" rounds of clustering
    for _ in range(num_midsection_rounds):
        round_idx += 1
        timer.init_timing(f"round-{round_idx}")
        console.print(f"(Midsection) Round {round_idx}: Re-clustering in chunks")

        file_pairs = _get_prev_round_centroids_and_n(out_dir, round_idx, console)

        batches = _chunk_file_pairs_in_batches(file_pairs, bin_size, console)

        merging_fn = _TreeMergingRound(
            round_idx=round_idx,
            all_fp_paths=input_files,
            merge_criterion=common_kwargs["merge_criterion"],
            threshold=threshold + midsection_threshold_change,
            branching_factor=common_kwargs["branching_factor"],
            out_dir=common_kwargs["out_dir"],
            reclustering_iterations=reclustering_iterations,
            extra_threshold=extra_threshold,
            reweight_centroids=reweight_centroids,
        )
        num_ps = min(num_midsection_processes, len(batches))
        console.print(f"    - Processing {len(batches)} inputs with {num_ps} processes")
        with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
            if num_ps == 1:
                for batch_info in batches:
                    merging_fn(batch_info)
            else:
                with mp_context.Pool(
                    processes=num_ps, maxtasksperchild=max_tasks_per_process
                ) as pool:
                    pool.map(merging_fn, batches)

        timer.end_timing(f"round-{round_idx}", console)
        console.print_peak_mem(out_dir)

    ###########################################
    # Final "Tree-Merging" round of clustering
    round_idx += 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Final) Round {round_idx}: Final round of clustering")
    file_pairs = _get_prev_round_centroids_and_n(out_dir, round_idx, console)

    final_fn = _FinalTreeMergingRound(
        save_tree=save_tree,
        save_centroids=save_centroids,
        merge_criterion=common_kwargs["merge_criterion"],
        branching_factor=common_kwargs["branching_factor"],
        threshold=threshold + midsection_threshold_change,
        out_dir=common_kwargs["out_dir"],
    )
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        final_fn(("", file_pairs))

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(out_dir)
    # Remove intermediate files
    if cleanup:
        for f in out_dir.glob("round-*.npy"):
            f.unlink()
        for f in out_dir.glob("round-*.pkl"):
            f.unlink()
    console.print()
    timer.end_timing("total", console, indent=False)
    return timer
