use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedSemType, ArchivedTableType,
    Offsets, SemType, TableInfo,
};
use clap::Parser;
#[cfg(feature = "vecdb")]
use faiss::index::io::{IoFlags, read_index_with_flags};
#[cfg(feature = "vecdb")]
use faiss::index::{Index as FaissIndex, IndexImpl};
use half::bf16;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::izip;
use memmap2::{Mmap, MmapOptions};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::index;
use rayon::prelude::*;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use std::alloc;
use std::collections::{HashMap, HashSet};

use std::fs;
use std::io::{BufReader, Read};
use std::str;
#[cfg(feature = "vecdb")]
use std::sync::Mutex;
use std::time::{Duration, Instant};

unsafe fn alloc_zeroed_vec<T>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    let layout = alloc::Layout::array::<T>(len).unwrap();
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) }
}

const MAX_F2P_NBRS: usize = 5;

fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push('_');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

#[derive(Debug)]
enum BuildError {
    MissingTargetCol,
    NanTargetValue,
}

#[inline]
fn check_deadline(deadline: Instant) {
    if Instant::now() >= deadline {
        panic!("timeout_per_item exceeded");
    }
}

const DEADLINE_CHECK_EVERY: usize = 1024;

const DRAW_BUDGET: usize = 2;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    timestamps: Vec<i32>,

    seed_node_idxs: Vec<i32>,
    bfs_depths: Vec<i32>,

    batch_mask: Vec<bool>,
    seq_len: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
    timestamps: &'a mut [i32],
    seed_node_idxs: &'a mut [i32],
    bfs_depths: &'a mut [i32],
}

impl Vecs {
    fn new(bs: usize, seq_len: usize, d_text: usize) -> Self {
        let l = bs * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: unsafe { alloc_zeroed_vec(l * d_text) },
            sem_types: vec![0; l],
            number_values: unsafe { alloc_zeroed_vec(l) },
            text_values: unsafe { alloc_zeroed_vec(l * d_text) },
            datetime_values: unsafe { alloc_zeroed_vec(l) },
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            timestamps: vec![i32::MIN; l],
            seed_node_idxs: vec![-1; l],
            bfs_depths: vec![-1; l],

            batch_mask: vec![true; bs],
            seq_len,
        }
    }

    fn chunks_exact_mut(
        &mut self,
        seq_len: usize,
        d_text: usize,
    ) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
            self.timestamps.chunks_exact_mut(seq_len),
            self.seed_node_idxs.chunks_exact_mut(seq_len),
            self.bfs_depths.chunks_exact_mut(seq_len)
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
                seed_node_idxs,
                bfs_depths,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
                seed_node_idxs,
                bfs_depths,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("timestamps", PyArray1::from_vec(py, self.timestamps))
                .into_py_any(py)
                .unwrap(),
            (
                "seed_node_idxs",
                PyArray1::from_vec(py, self.seed_node_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("bfs_depths", PyArray1::from_vec(py, self.bfs_depths))
                .into_py_any(py)
                .unwrap(),
            ("batch_mask", PyArray1::from_vec(py, self.batch_mask))
                .into_py_any(py)
                .unwrap(),
            ("seq_len", self.seq_len).into_py_any(py).unwrap(),
        ])
    }
}

#[cfg(feature = "vecdb")]
struct VectorDbEntry {
    node_idx_offset: i32,
    dim: usize,
    num_rows: usize,
    vectors_mmap: Mmap,
    index: Mutex<IndexImpl>,
}

#[cfg(feature = "vecdb")]
unsafe impl Send for VectorDbEntry {}
#[cfg(feature = "vecdb")]
unsafe impl Sync for VectorDbEntry {}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    offsets: Vec<i64>,
    table_info: HashMap<String, TableInfo>,

    #[cfg(feature = "vecdb")]
    vector_db: Option<HashMap<String, VectorDbEntry>>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
    table_name: String,
}

#[pyclass]
pub struct Sampler {
    global_rank: usize,
    local_rank: usize,
    world_size: usize,
    datasets: HashMap<String, Dataset>,
    items: Vec<Item>,
    local_ctx_size_list: Vec<usize>,
    bfs_width_list: Vec<usize>,

    num_walks: usize,
    walk_length: usize,

    prefer_latest_list: Vec<bool>,
    mask_prob_max: f64,
    step: u64,
    stride: u64,
    d_text: usize,

    shuffle_seed: u64,

    context_seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,

    cutoff_timestamps: Vec<Option<i32>>,
    items_per_task: i64,
    dataset_tuples: Vec<(String, String, i32, i32)>,
    table_ranges: Vec<(i32, i32)>,
    quiet: bool,

    legacy_boolean: bool,

    timeout_per_item: f64,

    #[cfg_attr(not(feature = "vecdb"), allow(dead_code))]
    vector_db_path: Option<String>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dataset_tuples: Vec<(String, String, i32, i32)>,
        global_rank: usize,
        local_rank: usize,
        world_size: usize,
        local_ctx_size_list: Vec<usize>,
        bfs_width_list: Vec<usize>,
        num_walks: usize,
        walk_length: usize,
        prefer_latest_list: Vec<bool>,
        mask_prob_max: f64,
        embedder: &str,
        pre_dir: String,
        d_text: usize,
        shuffle_seed: u64,
        context_seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        cutoff_timestamps: Vec<Option<i32>>,
        items_per_task: i64,
        quiet: bool,
        ignore_data_errors: bool,
        num_prev_skipped: usize,
        mmap_populate: bool,
        legacy_boolean: bool,
        timeout_per_item: f64,
        vector_db_path: Option<String>,
    ) -> Self {
        py.allow_threads(|| {
            Self::new_impl(
                dataset_tuples,
                global_rank,
                local_rank,
                world_size,
                local_ctx_size_list,
                bfs_width_list,
                num_walks,
                walk_length,
                prefer_latest_list,
                mask_prob_max,
                embedder,
                pre_dir,
                d_text,
                shuffle_seed,
                context_seed,
                target_columns,
                columns_to_drop,
                cutoff_timestamps,
                items_per_task,
                quiet,
                ignore_data_errors,
                num_prev_skipped,
                mmap_populate,
                legacy_boolean,
                timeout_per_item,
                vector_db_path,
            )
        })
    }

    #[getter]
    fn num_items(&self) -> usize {
        self.items.len()
    }

    fn batch_py(
        &mut self,
        py: Python<'_>,
        batch_idx: Option<usize>,
        bs: usize,
        ctx_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let vecs = match batch_idx {
            Some(idx) => self.batch(Some(idx), 0, bs, ctx_size),
            None => {
                let step = self.step;
                let r = self.batch(None, step, bs, ctx_size);
                self.step += self.stride;
                r
            }
        };
        vecs.into_pyobject(py)
    }

    fn batch_for_nodes_py(
        &self,
        py: Python<'_>,
        node_idxs: Vec<i32>,
        dataset_idx: usize,
        ctx_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let bs = node_idxs.len();
        let table_name = &self.dataset_tuples[dataset_idx].1;
        let mut vecs = Vecs::new(bs, ctx_size, self.d_text);

        vecs.chunks_exact_mut(ctx_size, self.d_text)
            .enumerate()
            .par_bridge()
            .for_each(|(i, slices)| {
                let item = Item {
                    dataset_idx: dataset_idx as i32,
                    node_idx: node_idxs[i],
                    table_name: table_name.clone(),
                };
                self.seq(&item, i, slices, 0, ctx_size);
            });

        vecs.into_pyobject(py)
    }

    fn set_step_py(&mut self, step: u64) {
        self.step = step;
    }

    fn set_stride_py(&mut self, stride: u64) {
        self.stride = stride;
    }

    fn set_mask_prob_max_py(&mut self, mask_prob_max: f64) {
        self.mask_prob_max = mask_prob_max;
    }

    #[getter]
    fn local_ctx_size(&self) -> usize {
        self.local_ctx_size_list[0]
    }

    #[getter]
    fn d_text(&self) -> usize {
        self.d_text
    }
}

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        dataset_tuples: Vec<(String, String, i32, i32)>,
        global_rank: usize,
        local_rank: usize,
        world_size: usize,
        local_ctx_size_list: Vec<usize>,
        bfs_width_list: Vec<usize>,
        num_walks: usize,
        walk_length: usize,
        prefer_latest_list: Vec<bool>,
        mask_prob_max: f64,
        embedder: &str,
        pre_dir: String,
        d_text: usize,
        shuffle_seed: u64,
        context_seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        cutoff_timestamps: Vec<Option<i32>>,
        items_per_task: i64,
        quiet: bool,
        ignore_data_errors: bool,
        num_prev_skipped: usize,
        mmap_populate: bool,
        legacy_boolean: bool,
        timeout_per_item: f64,
        vector_db_path: Option<String>,
    ) -> Self {
        let embedder_ref = embedder;

        assert_eq!(
            cutoff_timestamps.len(),
            dataset_tuples.len(),
            "cutoff_timestamps must have one entry per task"
        );

        let mut db_to_tables: HashMap<String, Vec<String>> = HashMap::new();
        for (db_name, table_name, _, _) in dataset_tuples.iter() {
            let entry = db_to_tables.entry(db_name.clone()).or_default();
            if !entry.iter().any(|t| t == table_name) {
                entry.push(table_name.clone());
            }
        }

        let mut mmap_opts = MmapOptions::new();
        if mmap_populate {
            mmap_opts.populate();
        }

        let pb = make_pb(
            db_to_tables.len() as u64,
            "loading databases",
            local_rank == 0 && !quiet,
        );

        let vector_db_path_ref = vector_db_path.as_deref();

        let try_load = |db_name: &String, table_names: &Vec<String>| {
            let compute = std::panic::AssertUnwindSafe(|| {
                Self::load_dataset(
                    db_name,
                    table_names,
                    &pre_dir,
                    &mmap_opts,
                    embedder_ref,
                    vector_db_path_ref,
                )
            });
            let r = if ignore_data_errors {
                std::panic::catch_unwind(compute).ok()
            } else {
                Some(compute())
            };
            pb.inc(1);
            if r.is_none() && local_rank == 0 && !quiet {
                eprintln!(
                    "\n\x1b[31mskipping db {}: cannot load its preprocessed files \
                     from {}\x1b[0m",
                    db_name, pre_dir,
                );
            }
            r
        };
        let datasets: HashMap<String, Dataset> = if db_to_tables.len() > 1 {
            db_to_tables
                .par_iter()
                .filter_map(|(db_name, table_names)| try_load(db_name, table_names))
                .collect()
        } else {
            db_to_tables
                .iter()
                .filter_map(|(db_name, table_names)| try_load(db_name, table_names))
                .collect()
        };
        pb.finish_and_clear();

        let (
            dataset_tuples,
            target_columns,
            columns_to_drop,
            cutoff_timestamps,
            table_ranges,
            rust_skipped,
        ) = {
            let mut kept_tuples: Vec<(String, String, i32, i32)> = Vec::new();
            let mut kept_targets: Vec<i32> = Vec::new();
            let mut kept_drops: Vec<Vec<i32>> = Vec::new();
            let mut kept_cutoffs: Vec<Option<i32>> = Vec::new();
            let mut kept_ranges: Vec<(i32, i32)> = Vec::new();
            let mut skipped: usize = 0;
            for (i, tuple) in dataset_tuples.into_iter().enumerate() {
                let (ref db_name, ref table_name, _, _) = tuple;
                let datasets_ref = &datasets;
                let db = db_name.clone();
                let table = table_name.clone();
                let compute = std::panic::AssertUnwindSafe(|| {
                    let dataset = datasets_ref.get(&db).unwrap_or_else(|| {
                        panic!("db {} was dropped (its files could not be loaded)", db)
                    });
                    let mut range_start = i32::MAX;
                    let mut range_end = i32::MIN;
                    for (key, info) in &dataset.table_info {
                        if let Some(colon_pos) = key.rfind(':')
                            && &key[..colon_pos] == table.as_str()
                        {
                            range_start = range_start.min(info.node_idx_offset);
                            range_end = range_end.max(info.node_idx_offset + info.num_nodes);
                        }
                    }
                    assert!(
                        range_start < range_end,
                        "table {} has no usable rows in db {}",
                        table,
                        db,
                    );
                    (range_start, range_end)
                });
                let result = if ignore_data_errors {
                    std::panic::catch_unwind(compute)
                } else {
                    Ok(compute())
                };
                match result {
                    Ok(range) => {
                        kept_tuples.push(tuple);
                        kept_targets.push(target_columns[i]);
                        kept_drops.push(columns_to_drop[i].clone());
                        kept_cutoffs.push(cutoff_timestamps[i]);
                        kept_ranges.push(range);
                    }
                    Err(panic_info) => {
                        if local_rank == 0 && !quiet {
                            let msg = panic_info
                                .downcast_ref::<String>()
                                .map(|s| s.as_str())
                                .or_else(|| panic_info.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic");
                            eprintln!(
                                "\n\x1b[31mskipping task {}/{}: {}\x1b[0m",
                                db_name, table_name, msg
                            );
                        }
                        skipped += 1;
                    }
                }
            }
            (
                kept_tuples,
                kept_targets,
                kept_drops,
                kept_cutoffs,
                kept_ranges,
                skipped,
            )
        };
        assert!(
            !dataset_tuples.is_empty(),
            "All tasks were skipped due to errors, cannot proceed."
        );
        assert!(
            !local_ctx_size_list.is_empty(),
            "local_ctx_size_list must be non-empty"
        );
        assert!(
            !bfs_width_list.is_empty(),
            "bfs_width_list must be non-empty"
        );
        assert!(
            !prefer_latest_list.is_empty(),
            "prefer_latest_list must be non-empty"
        );
        let mut sampler = Self {
            global_rank,
            local_rank,
            world_size,
            datasets,
            items: Vec::new(),
            local_ctx_size_list,
            bfs_width_list,
            num_walks,
            walk_length,
            prefer_latest_list,
            mask_prob_max,
            step: 0,
            stride: 1,
            d_text,
            shuffle_seed: StdRng::seed_from_u64(shuffle_seed).random(),
            context_seed: StdRng::seed_from_u64(context_seed).random(),
            target_columns,
            columns_to_drop,
            cutoff_timestamps,
            items_per_task,
            dataset_tuples,
            table_ranges,
            quiet,
            legacy_boolean,
            timeout_per_item,
            vector_db_path,
        };
        sampler.create_items();
        if sampler.local_rank == 0 && !sampler.quiet {
            let num_dbs = sampler
                .dataset_tuples
                .iter()
                .map(|(db, _, _, _)| db)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let num_tasks = sampler.dataset_tuples.len();
            let num_items = sampler.items.len();
            let num_skipped = num_prev_skipped + rust_skipped;
            println!(
                "\ndata stats: \x1b[1m{}\x1b[0m dbs, \x1b[1m{}\x1b[0m tasks, \x1b[1m{}\x1b[0m items, \x1b[1m{}\x1b[0m skipped",
                fmt_thousands(num_dbs),
                fmt_thousands(num_tasks),
                fmt_thousands(num_items),
                fmt_thousands(num_skipped),
            );
        }
        sampler
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "vecdb"), allow(unused_variables))]
    fn load_dataset(
        db_name: &str,
        table_names: &[String],
        pre_dir: &str,
        mmap_opts: &MmapOptions,
        embedder_ref: &str,
        vector_db_path: Option<&str>,
    ) -> (String, Dataset) {
        let pre_path = format!("{}/{}", pre_dir, db_name);

        let nodes_path = format!("{}/nodes.rkyv", pre_path);
        let file = fs::File::open(&nodes_path).unwrap();
        let mmap = unsafe { mmap_opts.map(&file).unwrap() };

        let text_path = format!("{}/text_emb_{}.bin", pre_path, embedder_ref);
        let text_file = fs::File::open(&text_path).unwrap();
        let text_mmap = unsafe { mmap_opts.map(&text_file).unwrap() };

        let offsets_path = format!("{}/offsets.rkyv", pre_path);
        let file = fs::File::open(&offsets_path).unwrap();
        let mut bytes = Vec::new();
        BufReader::new(file).read_to_end(&mut bytes).unwrap();
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
        let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
        let offsets = offsets.offsets;

        let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
        let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
        let p2f_adj_mmap = unsafe { mmap_opts.map(&p2f_adj_file).unwrap() };

        let table_info_path = format!("{}/table_info.json", pre_path);
        let table_info_file = fs::File::open(&table_info_path).unwrap();
        let table_info: HashMap<String, TableInfo> =
            serde_json::from_reader(BufReader::new(table_info_file)).unwrap();

        #[cfg(not(feature = "vecdb"))]
        assert!(
            vector_db_path.is_none(),
            "vector_db_path is set but rustler was built without the 'vecdb' \
             feature; rebuild with `--features vecdb`"
        );

        #[cfg(feature = "vecdb")]
        let vector_db = vector_db_path.map(|root| {
            let mut entries: HashMap<String, VectorDbEntry> = HashMap::new();
            for table in table_names {
                let index_path = format!("{}/{}/{}.index", root, db_name, table);
                let vectors_path = format!("{}/{}/{}_vectors.bin", root, db_name, table);

                let node_idx_offset = table_info
                    .iter()
                    .filter(|(key, _)| key.rsplit_once(':').map(|(t, _)| t) == Some(table.as_str()))
                    .map(|(_, info)| info.node_idx_offset)
                    .min()
                    .unwrap_or_else(|| {
                        panic!("table {} not found in table_info for db {}", table, db_name,)
                    });

                let faiss_index =
                    read_index_with_flags(&index_path, IoFlags::MEM_MAP | IoFlags::READ_ONLY)
                        .unwrap_or_else(|e| {
                            panic!("failed to load FAISS index {}: {}", index_path, e)
                        });
                let dim = faiss_index.d() as usize;

                let vecs_file = fs::File::open(&vectors_path).unwrap_or_else(|e| {
                    panic!("failed to open vectors file {}: {}", vectors_path, e)
                });
                let vectors_mmap = unsafe { mmap_opts.map(&vecs_file).unwrap() };
                let bytes = vectors_mmap.len();
                assert!(
                    bytes % (dim * std::mem::size_of::<f32>()) == 0,
                    "vectors file {} size ({} bytes) is not a multiple of dim*4={}",
                    vectors_path,
                    bytes,
                    dim * std::mem::size_of::<f32>(),
                );
                let num_rows = bytes / (dim * std::mem::size_of::<f32>());

                entries.insert(
                    table.clone(),
                    VectorDbEntry {
                        node_idx_offset,
                        dim,
                        num_rows,
                        vectors_mmap,
                        index: Mutex::new(faiss_index),
                    },
                );
            }
            entries
        });

        (
            db_name.to_owned(),
            Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
                table_info,
                #[cfg(feature = "vecdb")]
                vector_db,
            },
        )
    }

    fn create_items(&mut self) {
        self.items.clear();

        let pb = make_pb(
            self.dataset_tuples.len() as u64,
            "subsampling tasks",
            self.local_rank == 0 && !self.quiet,
        );

        for (i, &(_, ref table_name, node_idx_offset, num_nodes)) in
            self.dataset_tuples.iter().enumerate()
        {
            let num_to_sample = if self.items_per_task == -1 {
                num_nodes as usize
            } else {
                (num_nodes as usize).min(self.items_per_task as usize)
            };

            let rng_seed = self.shuffle_seed + (i as u64);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            let sampled_indices = index::sample(&mut rng, num_nodes as usize, num_to_sample);

            for idx in sampled_indices.iter() {
                let node_idx = node_idx_offset + idx as i32;
                self.items.push(Item {
                    dataset_idx: i as i32,
                    node_idx,
                    table_name: table_name.clone(),
                });
            }
            pb.inc(1);
        }
        pb.finish_and_clear();

        let mut rng = StdRng::seed_from_u64(self.shuffle_seed);
        self.items.shuffle(&mut rng);
    }

    fn len(&self, bs: usize) -> usize {
        self.items.len().div_ceil(bs * self.world_size)
    }

    fn batch(&self, batch_idx: Option<usize>, step: u64, bs: usize, ctx_size: usize) -> Vecs {
        match batch_idx {
            Some(idx) => {
                let offset = self.global_rank * bs + idx * bs * self.world_size;
                let mut vecs = Vecs::new(bs, ctx_size, self.d_text);

                for (i, m) in vecs.batch_mask.iter_mut().enumerate() {
                    *m = offset + i < self.items.len();
                }

                let timeout = Duration::from_secs_f64(self.timeout_per_item);
                let timed_out: Vec<bool> = vecs
                    .chunks_exact_mut(ctx_size, self.d_text)
                    .enumerate()
                    .map(|(i, mut slices)| {
                        let j = offset + i;
                        if j >= self.items.len() {


                            return false;
                        }
                        let item = &self.items[j];
                        let deadline = Instant::now() + timeout;
                        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || self.seq_build(item, &mut slices, 0, ctx_size, deadline),
                        ));
                        let failed = !matches!(caught, Ok(Ok(())));
                        if failed {

                            slices.is_targets.fill(false);
                            slices.is_padding.fill(true);
                            let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
                            eprintln!(
                                "\n\x1b[31meval: dropping item (db={}, table={}, node_idx={}): build failed/timed out\x1b[0m",
                                db_name, item.table_name, item.node_idx
                            );
                        }
                        failed
                    })
                    .collect();
                for (i, &failed) in timed_out.iter().enumerate() {
                    if failed {
                        vecs.batch_mask[i] = false;
                    }
                }
                vecs
            }
            None => {
                let mut rng = StdRng::seed_from_u64(
                    (self.context_seed + step).wrapping_add(0xE0E0_E0E0_E0E0_E0E0),
                );

                let mut vecs = Vecs::new(bs, ctx_size, self.d_text);

                vecs.batch_mask.iter_mut().for_each(|m| *m = true);

                let global_bs = bs * self.world_size;
                let global_indices: Vec<usize> = (0..global_bs)
                    .map(|_| rng.random_range(0..self.items.len()))
                    .collect();

                vecs.chunks_exact_mut(ctx_size, self.d_text)
                    .enumerate()
                    .for_each(|(i, slices)| {
                        let j = self.global_rank * bs + i;
                        let item = &self.items[global_indices[j]];
                        self.seq(item, global_indices[j], slices, step, ctx_size);
                    });
                vecs
            }
        }
    }

    fn seq(&self, item: &Item, item_idx: usize, mut slices: Slices, step: u64, ctx_len: usize) {
        let mut current_item = item;
        let mut retry_seed = item_idx as u64;
        let timeout = Duration::from_secs_f64(self.timeout_per_item);
        loop {
            let deadline = Instant::now() + timeout;
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.seq_build(current_item, &mut slices, step, ctx_len, deadline)
            }));
            match caught {
                Ok(Ok(())) => break,
                Ok(Err(_)) => {}
                Err(panic_info) => {
                    let msg = panic_info
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic_info.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    let db_name = &self.dataset_tuples[current_item.dataset_idx as usize].0;
                    let target_column = self.target_columns[current_item.dataset_idx as usize];
                    eprintln!(
                        "\n\x1b[31mskipping item {item_idx} (db={}, table={}, target_node_idx={}, target_column_idx={}, step={step}, ctx_len={ctx_len}): {msg}\x1b[0m",
                        db_name, current_item.table_name, current_item.node_idx, target_column
                    );
                }
            }
            slices.is_targets.fill(false);
            slices.is_padding.fill(true);
            let next_idx = StdRng::seed_from_u64(retry_seed).random_range(0..self.items.len());
            current_item = &self.items[next_idx];
            retry_seed = next_idx as u64;
        }
    }

    fn seq_build(
        &self,
        item: &Item,
        slices: &mut Slices,
        step: u64,
        ctx_len: usize,
        deadline: Instant,
    ) -> Result<(), BuildError> {
        check_deadline(deadline);
        let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
        let dataset = &self.datasets[db_name];
        let target_column = self.target_columns[item.dataset_idx as usize];
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];
        let cutoff = self.cutoff_timestamps[item.dataset_idx as usize];

        let target_node_idx = item.node_idx;
        let target_node = get_node(dataset, target_node_idx);

        let target_cell_i = match target_node
            .col_name_idxs
            .iter()
            .position(|&col_idx| i32::from(col_idx) == target_column)
        {
            Some(i) => i,
            None => return Err(BuildError::MissingTargetCol),
        };

        let target_value_is_nan = {
            let sem = &target_node.sem_types[target_cell_i];
            match sem {
                ArchivedSemType::Number => {
                    f32::from(target_node.number_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::DateTime => {
                    f32::from(target_node.datetime_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Boolean => {
                    f32::from(target_node.boolean_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Text => false,
            }
        };
        if target_value_is_nan {
            return Err(BuildError::NanTargetValue);
        }

        let step_seed: u64 = StdRng::seed_from_u64(self.context_seed + step).random();

        let mut seq_rng = StdRng::seed_from_u64(step_seed + target_node_idx as u64);
        let mask_prob = seq_rng.random::<f64>() * self.mask_prob_max;

        let valid_local_ctx_size_list: Vec<usize> = self
            .local_ctx_size_list
            .iter()
            .copied()
            .filter(|&l| l <= ctx_len)
            .collect();
        assert!(
            !valid_local_ctx_size_list.is_empty(),
            "No local_ctx_size in {:?} is <= ctx_len={}",
            self.local_ctx_size_list,
            ctx_len
        );
        let local_ctx_size =
            valid_local_ctx_size_list[seq_rng.random_range(0..valid_local_ctx_size_list.len())];
        let bfs_width = self.bfs_width_list[seq_rng.random_range(0..self.bfs_width_list.len())];
        let prefer_latest =
            self.prefer_latest_list[seq_rng.random_range(0..self.prefer_latest_list.len())];

        #[cfg(feature = "vecdb")]
        let use_vector_db = self.vector_db_path.is_some();
        #[cfg(not(feature = "vecdb"))]
        let use_vector_db = false;
        let visit_counts = if !use_vector_db && self.num_walks > 0 {
            self.compute_visit_counts(
                dataset,
                target_node_idx,
                target_node,
                self.num_walks,
                self.walk_length,
                step_seed,
                cutoff,
                deadline,
            )
        } else {
            HashMap::new()
        };

        let mut visited_sorted: Vec<i32> = visit_counts.keys().copied().collect();
        let priority: HashMap<i32, u64> = visited_sorted
            .iter()
            .map(|&n| {
                (
                    n,
                    StdRng::seed_from_u64(step_seed.wrapping_add(n as u64)).random::<u64>(),
                )
            })
            .collect();
        check_deadline(deadline);
        if prefer_latest {
            let ts_of: HashMap<i32, Option<i32>> = visited_sorted
                .iter()
                .map(|&n| {
                    let ts = get_node(dataset, n).timestamp.as_ref().map(|t| (*t).into());
                    (n, ts)
                })
                .collect();
            visited_sorted.sort_by(|a, b| {
                ts_of[b]
                    .cmp(&ts_of[a])
                    .then_with(|| visit_counts[b].cmp(&visit_counts[a]))
                    .then_with(|| priority[a].cmp(&priority[b]))
            });
        } else {
            visited_sorted.sort_by(|a, b| {
                visit_counts[b]
                    .cmp(&visit_counts[a])
                    .then_with(|| priority[a].cmp(&priority[b]))
            });
        }
        check_deadline(deadline);

        let (range_start, range_end) = self.table_ranges[item.dataset_idx as usize];
        let total_table = (range_end - range_start) as usize;
        let fallback_seed = step_seed
            .wrapping_add(target_node_idx as u64)
            .wrapping_add(0xA5A5_A5A5_A5A5_A5A5);

        let mut visited_in_ctx: HashSet<i32> = HashSet::with_capacity(ctx_len);
        let mut visited_at_depth: HashMap<i32, usize> = HashMap::with_capacity(ctx_len);

        let mut cells_to_add: Vec<(i32, usize, i32, i32, i32)> = Vec::with_capacity(ctx_len);

        let mut bfs_rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0xB0B0_B0B0_B0B0_B0B0),
        );

        cells_to_add.push((
            target_node_idx,
            target_cell_i,
            target_column,
            target_node_idx,
            0,
        ));

        'fill_ctx: {
            if extend_with_seed_bfs(
                self,
                dataset,
                target_node_idx,
                target_node_idx,
                target_node,
                target_column,
                columns_to_drop,
                cutoff,
                local_ctx_size,
                bfs_width,
                ctx_len,
                &mut bfs_rng,
                &mut visited_at_depth,
                &mut visited_in_ctx,
                &mut cells_to_add,
                deadline,
            ) {
                break 'fill_ctx;
            }

            let mut tier1_seen: HashSet<i32> = HashSet::new();
            if use_vector_db {
                #[cfg(feature = "vecdb")]
                {
                    let entry = dataset
                        .vector_db
                        .as_ref()
                        .expect("vector_db_path is set but dataset.vector_db was not loaded")
                        .get(item.table_name.as_str())
                        .unwrap_or_else(|| {
                            panic!(
                                "vector_db has no FAISS entry for table '{}'",
                                item.table_name
                            )
                        });
                    let mut vdb = VectorDbStream::new(entry, target_node_idx, target_node, cutoff);
                    while let Some(seed_node_idx) = vdb.next(dataset) {
                        check_deadline(deadline);
                        if seed_label_missing(get_node(dataset, seed_node_idx), target_column) {
                            continue;
                        }
                        if extend_with_seed_bfs(
                            self,
                            dataset,
                            seed_node_idx,
                            target_node_idx,
                            target_node,
                            target_column,
                            columns_to_drop,
                            cutoff,
                            local_ctx_size,
                            bfs_width,
                            ctx_len,
                            &mut bfs_rng,
                            &mut visited_at_depth,
                            &mut visited_in_ctx,
                            &mut cells_to_add,
                            deadline,
                        ) {
                            break 'fill_ctx;
                        }
                    }
                }
            } else {
                for &seed_node_idx in visited_sorted.iter() {
                    check_deadline(deadline);
                    if seed_label_missing(get_node(dataset, seed_node_idx), target_column) {
                        continue;
                    }
                    if extend_with_seed_bfs(
                        self,
                        dataset,
                        seed_node_idx,
                        target_node_idx,
                        target_node,
                        target_column,
                        columns_to_drop,
                        cutoff,
                        local_ctx_size,
                        bfs_width,
                        ctx_len,
                        &mut bfs_rng,
                        &mut visited_at_depth,
                        &mut visited_in_ctx,
                        &mut cells_to_add,
                        deadline,
                    ) {
                        break 'fill_ctx;
                    }
                }
            }
            if use_vector_db {
                break 'fill_ctx;
            }
            for &n in visit_counts.keys() {
                tier1_seen.insert(n);
            }

            if total_table == 0 {
                break 'fill_ctx;
            }

            let remaining_cells = ctx_len.saturating_sub(cells_to_add.len());
            let fallback_amount = std::cmp::min(remaining_cells, total_table);
            if fallback_amount == 0 {
                break 'fill_ctx;
            }

            let pool_size = if prefer_latest {
                std::cmp::min(
                    total_table,
                    fallback_amount.saturating_mul(PREFER_LATEST_OVERSAMPLE),
                )
            } else {
                fallback_amount
            };
            let mut fallback_rng = StdRng::seed_from_u64(fallback_seed);
            let fallback_offsets = index::sample(&mut fallback_rng, total_table, pool_size);
            check_deadline(deadline);
            let mut fallback_order: Vec<i32> = fallback_offsets
                .iter()
                .map(|off| range_start + off as i32)
                .collect();
            if prefer_latest {
                let bound =
                    temporal_bound(target_node.timestamp.as_ref().map(|t| (*t).into()), cutoff);
                fallback_order.retain(|&n| {
                    if n == target_node_idx || tier1_seen.contains(&n) {
                        return false;
                    }
                    let node = get_node(dataset, n);
                    !past_bound(node, bound)
                        && !same_horizon_task_row(node, target_node)
                        && !seed_label_missing(node, target_column)
                });
                check_deadline(deadline);
                let ts_of: HashMap<i32, Option<i32>> = fallback_order
                    .iter()
                    .map(|&n| {
                        (
                            n,
                            get_node(dataset, n).timestamp.as_ref().map(|t| (*t).into()),
                        )
                    })
                    .collect();
                let prio: HashMap<i32, u64> = fallback_order
                    .iter()
                    .map(|&n| {
                        (
                            n,
                            StdRng::seed_from_u64(step_seed.wrapping_add(n as u64)).random::<u64>(),
                        )
                    })
                    .collect();
                fallback_order
                    .sort_by(|a, b| ts_of[b].cmp(&ts_of[a]).then_with(|| prio[a].cmp(&prio[b])));
                fallback_order.truncate(fallback_amount);
            }
            for seed_node_idx in fallback_order {
                check_deadline(deadline);
                if seed_node_idx == target_node_idx {
                    continue;
                }
                if tier1_seen.contains(&seed_node_idx) {
                    continue;
                }

                let seed_node = get_node(dataset, seed_node_idx);

                if past_bound(
                    seed_node,
                    temporal_bound(target_node.timestamp.as_ref().map(|t| (*t).into()), cutoff),
                ) || same_horizon_task_row(seed_node, target_node)
                {
                    continue;
                }
                if seed_label_missing(seed_node, target_column) {
                    continue;
                }
                if extend_with_seed_bfs(
                    self,
                    dataset,
                    seed_node_idx,
                    target_node_idx,
                    target_node,
                    target_column,
                    columns_to_drop,
                    cutoff,
                    local_ctx_size,
                    bfs_width,
                    ctx_len,
                    &mut bfs_rng,
                    &mut visited_at_depth,
                    &mut visited_in_ctx,
                    &mut cells_to_add,
                    deadline,
                ) {
                    break 'fill_ctx;
                }
            }
        }

        let mut rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0xC0C0_C0C0_C0C0_C0C0),
        );
        let mut seq_i = 0;
        let mut last_node_idx: i32 = -1;
        let mut cached_node: Option<&ArchivedNode> = None;
        for &(node_idx, cell_i, _col_idx, seed_node_idx, depth) in cells_to_add.iter() {
            check_deadline(deadline);
            if seq_i >= ctx_len {
                break;
            }

            if node_idx != last_node_idx {
                cached_node = Some(get_node(dataset, node_idx));
                last_node_idx = node_idx;
            }

            self.add_single_cell(
                dataset,
                cached_node.unwrap(),
                cell_i,
                target_node_idx,
                target_column,
                &mut rng,
                &mut seq_i,
                slices,
                mask_prob,
                seed_node_idx,
                depth,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_visit_counts(
        &self,
        dataset: &Dataset,
        source_idx: i32,
        source_node: &ArchivedNode,
        num_walks: usize,
        max_walk_length: usize,
        step_seed: u64,
        cutoff: Option<i32>,
        deadline: Instant,
    ) -> HashMap<i32, usize> {
        let mut rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(source_idx as u64)
                .wrapping_add(0xD0D0_D0D0_D0D0_D0D0),
        );

        let bound = temporal_bound(source_node.timestamp.as_ref().map(|t| (*t).into()), cutoff);

        let target_ts: Option<i32> = source_node.timestamp.as_ref().map(|t| (*t).into());

        let mut similar_node_visits: HashMap<i32, usize> = HashMap::new();

        for _ in 0..num_walks {
            check_deadline(deadline);
            let mut current_idx = source_idx;

            for _ in 0..max_walk_length {
                let current_node = get_node(dataset, current_idx);

                if current_node.table_name_idx == source_node.table_name_idx
                    && current_idx != source_idx
                    && !past_bound(current_node, bound)
                    && !same_horizon_task_row(current_node, source_node)
                {
                    *similar_node_visits.entry(current_idx).or_insert(0) += 1;
                }

                let next_idx = match self.select_random_neighbor(
                    dataset,
                    current_idx,
                    bound,
                    target_ts,
                    &mut rng,
                ) {
                    Some(idx) => idx,
                    None => break,
                };

                current_idx = next_idx;
            }
        }

        similar_node_visits
    }

    fn select_random_neighbor(
        &self,
        dataset: &Dataset,
        current_idx: i32,
        bound: Option<i32>,
        target_ts: Option<i32>,
        rng: &mut StdRng,
    ) -> Option<i32> {
        let current_node = get_node(dataset, current_idx);
        let p2f_edges = get_p2f_edges(dataset, current_idx);

        let valid_p2f_count = if bound.is_some() {
            p2f_edges
                .as_slice()
                .partition_point(|edge| edge_visible(edge, bound, target_ts))
        } else {
            p2f_edges.len()
        };

        let f2p_edges = current_node.f2p_edges.as_slice();
        let valid_f2p_count = if bound.is_some() {
            f2p_edges
                .iter()
                .filter(|edge| edge_visible(edge, bound, target_ts))
                .count()
        } else {
            f2p_edges.len()
        };

        let total_valid_neighbors = valid_f2p_count + valid_p2f_count;
        if total_valid_neighbors == 0 {
            return None;
        }

        let rand_idx = rng.random_range(0..total_valid_neighbors);
        if rand_idx < valid_f2p_count {
            Some(
                f2p_edges
                    .iter()
                    .filter(|edge| edge_visible(edge, bound, target_ts))
                    .nth(rand_idx)
                    .unwrap()
                    .node_idx
                    .into(),
            )
        } else {
            Some(p2f_edges[rand_idx - valid_f2p_count].node_idx.into())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_single_cell(
        &self,
        dataset: &Dataset,
        node: &ArchivedNode,
        cell_i: usize,
        target_node_idx: i32,
        target_column: i32,
        rng: &mut StdRng,
        seq_i: &mut usize,
        slices: &mut Slices,
        mask_prob: f64,
        seed_node_idx: i32,
        bfs_depth: i32,
    ) {
        let value_is_nan = match &node.sem_types[cell_i] {
            ArchivedSemType::Number => f32::from(node.number_values[cell_i]).is_nan(),
            ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]).is_nan(),
            ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]).is_nan(),
            ArchivedSemType::Text => false,
        };
        if value_is_nan {
            return;
        }

        slices.node_idxs[*seq_i] = node.node_idx.into();

        assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
        for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
            slices.f2p_nbr_idxs[*seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
        }

        slices.table_name_idxs[*seq_i] = node.table_name_idx.into();
        slices.col_name_idxs[*seq_i] = node.col_name_idxs[cell_i].into();
        slices.class_value_idxs[*seq_i] = node.class_value_idx[cell_i].into();
        slices.col_name_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text].copy_from_slice(
            get_text_emb(dataset, slices.col_name_idxs[*seq_i], self.d_text),
        );

        slices.sem_types[*seq_i] = match node.sem_types[cell_i] {
            ArchivedSemType::Boolean if !self.legacy_boolean => SemType::Number as i32,
            ref sem => sem.clone() as i32,
        };
        slices.number_values[*seq_i] = bf16::from_f32(match node.sem_types[cell_i] {
            ArchivedSemType::Boolean => node.boolean_values[cell_i].into(),
            _ => node.number_values[cell_i].into(),
        });

        let text_idx: i32 = node.text_values[cell_i].into();
        slices.text_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

        slices.datetime_values[*seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

        slices.is_targets[*seq_i] =
            if node.node_idx == target_node_idx && node.col_name_idxs[cell_i] == target_column {
                true
            } else {
                rng.random::<f64>() < mask_prob
            };

        slices.is_task_nodes[*seq_i] =
            node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
        slices.is_padding[*seq_i] = false;
        slices.timestamps[*seq_i] = match node.timestamp.as_ref() {
            Some(ts) => (*ts).into(),
            None => i32::MIN,
        };
        slices.seed_node_idxs[*seq_i] = seed_node_idx;
        slices.bfs_depths[*seq_i] = bfs_depth;

        *seq_i += 1;
    }

    #[allow(clippy::too_many_arguments)]
    fn bfs_collect_nodes(
        &self,
        dataset: &Dataset,
        start_idx: i32,
        rng: &mut StdRng,
        cutoff: Option<i32>,
        target_ts: Option<i32>,
        local_ctx_size: usize,
        bfs_width: usize,
        visited_at_depth: &mut HashMap<i32, usize>,
        deadline: Instant,
    ) -> Vec<(i32, usize)> {
        let mut result: Vec<(i32, usize)> = Vec::with_capacity(128);

        let start_node = get_node(dataset, start_idx);
        let bound = temporal_bound(start_node.timestamp.as_ref().map(|t| (*t).into()), cutoff);
        let mut num_cells = 0;

        let mut f2p_ftr: Vec<(usize, i32)> = Vec::with_capacity(64);
        let mut p2f_ftr: Vec<Vec<i32>> = vec![vec![start_idx]];

        loop {
            check_deadline(deadline);

            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                match p2f_ftr.iter().position(|v| !v.is_empty()) {
                    None => return result,
                    Some(depth) => {
                        let r = rng.random_range(0..p2f_ftr[depth].len());
                        let l = p2f_ftr[depth].len();
                        p2f_ftr[depth].swap(r, l - 1);
                        let node_idx = p2f_ftr[depth].pop().unwrap();
                        (depth, node_idx)
                    }
                }
            };

            if let Some(&prev_depth) = visited_at_depth.get(&node_idx)
                && prev_depth <= depth
            {
                continue;
            }

            let node = get_node(dataset, node_idx);

            num_cells += node.col_name_idxs.len();
            if num_cells >= local_ctx_size {
                return result;
            }

            visited_at_depth.insert(node_idx, depth);

            result.push((node_idx, depth));

            for edge in node.f2p_edges.iter() {
                if !edge_visible(edge, bound, target_ts) {
                    continue;
                }
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            let p2f_edges = get_p2f_edges(dataset, node_idx);

            let valid_edges = if bound.is_some() {
                p2f_edges
                    .as_slice()
                    .partition_point(|edge| edge_visible(edge, bound, target_ts))
            } else {
                p2f_edges
                    .as_slice()
                    .partition_point(|edge| edge.timestamp.is_none())
            };

            let p2f_edges = &p2f_edges.as_slice()[..valid_edges];

            let eligible = |edge: &ArchivedEdge| {
                edge.table_type == ArchivedTableType::Db
                    || edge.table_name_idx == start_node.table_name_idx
            };
            let n = p2f_edges.len();
            let mut chosen: Vec<i32> = Vec::with_capacity(bfs_width.min(n));

            if n <= bfs_width {
                chosen.extend(
                    p2f_edges
                        .iter()
                        .filter(|e| eligible(e))
                        .map(|e| i32::from(e.node_idx)),
                );
            } else {
                let budget = (DRAW_BUDGET * bfs_width).min(n);
                let mut swapped: HashMap<usize, usize> = HashMap::with_capacity(budget);
                for j in 0..budget {
                    if chosen.len() == bfs_width {
                        break;
                    }
                    if j & (DEADLINE_CHECK_EVERY - 1) == 0 {
                        check_deadline(deadline);
                    }
                    let r = rng.random_range(j..n);
                    let pick = *swapped.get(&r).unwrap_or(&r);
                    let at_j = *swapped.get(&j).unwrap_or(&j);
                    swapped.insert(r, at_j);
                    if eligible(&p2f_edges[pick]) {
                        chosen.push(p2f_edges[pick].node_idx.into());
                    }
                }
            }

            for &node_idx in chosen.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(node_idx);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn extend_with_seed_bfs(
    sampler: &Sampler,
    dataset: &Dataset,
    seed_node_idx: i32,
    target_node_idx: i32,
    target_node: &ArchivedNode,
    target_column: i32,
    columns_to_drop: &[i32],
    cutoff: Option<i32>,
    local_ctx_size: usize,
    bfs_width: usize,
    ctx_len: usize,
    bfs_rng: &mut StdRng,
    visited_at_depth: &mut HashMap<i32, usize>,
    visited_in_ctx: &mut HashSet<i32>,
    cells_to_add: &mut Vec<(i32, usize, i32, i32, i32)>,
    deadline: Instant,
) -> bool {
    let bfs_nodes = sampler.bfs_collect_nodes(
        dataset,
        seed_node_idx,
        bfs_rng,
        cutoff,
        target_node.timestamp.as_ref().map(|t| (*t).into()),
        local_ctx_size,
        bfs_width,
        visited_at_depth,
        deadline,
    );

    for (bfs_node_idx, depth) in bfs_nodes {
        check_deadline(deadline);
        if visited_in_ctx.contains(&bfs_node_idx) {
            continue;
        }
        visited_in_ctx.insert(bfs_node_idx);

        let node = get_node(dataset, bfs_node_idx);

        if bfs_node_idx != target_node_idx && same_horizon_task_row(node, target_node) {
            continue;
        }

        for cell_i in 0..node.col_name_idxs.len() {
            if cell_i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                check_deadline(deadline);
            }
            let col_idx: i32 = node.col_name_idxs[cell_i].into();

            if columns_to_drop.contains(&col_idx)
                && (node.node_idx == target_node_idx
                    || (target_node.timestamp.is_some() && node.timestamp == target_node.timestamp))
            {
                continue;
            }

            if bfs_node_idx == target_node_idx && col_idx == target_column {
                continue;
            }

            cells_to_add.push((bfs_node_idx, cell_i, col_idx, seed_node_idx, depth as i32));

            if cells_to_add.len() == ctx_len {
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "vecdb")]
struct VectorDbStream<'a> {
    entry: &'a VectorDbEntry,
    target_node_idx: i32,
    target_ts: Option<i32>,
    query: &'a [f32],

    buffer: Vec<i32>,
    cursor: usize,

    last_k: usize,

    yielded: HashSet<i32>,
    exhausted: bool,
}

#[cfg(feature = "vecdb")]
const VDB_INIT_K: usize = 64;

#[cfg(feature = "vecdb")]
impl<'a> VectorDbStream<'a> {
    fn new(
        entry: &'a VectorDbEntry,
        target_node_idx: i32,
        target_node: &ArchivedNode,
        cutoff: Option<i32>,
    ) -> Self {
        let local_offset = target_node_idx - entry.node_idx_offset;
        assert!(
            local_offset >= 0 && (local_offset as usize) < entry.num_rows,
            "target node {} is outside the FAISS-indexed range [{}, {})",
            target_node_idx,
            entry.node_idx_offset,
            entry.node_idx_offset + entry.num_rows as i32,
        );
        let local_idx = local_offset as usize;
        let (pref, vectors, suf) = unsafe { entry.vectors_mmap.align_to::<f32>() };
        assert!(
            pref.is_empty() && suf.is_empty(),
            "FAISS vectors mmap is not f32-aligned"
        );
        let query = &vectors[local_idx * entry.dim..(local_idx + 1) * entry.dim];
        let target_ts = temporal_bound(target_node.timestamp.as_ref().map(|t| (*t).into()), cutoff);
        Self {
            entry,
            target_node_idx,
            target_ts,
            query,
            buffer: Vec::new(),
            cursor: 0,
            last_k: 0,
            yielded: HashSet::new(),
            exhausted: false,
        }
    }

    fn next(&mut self, dataset: &Dataset) -> Option<i32> {
        loop {
            if self.cursor < self.buffer.len() {
                let v = self.buffer[self.cursor];
                self.cursor += 1;
                return Some(v);
            }
            if self.exhausted {
                return None;
            }
            self.expand(dataset);
        }
    }

    fn expand(&mut self, dataset: &Dataset) {
        let new_k = if self.last_k == 0 {
            VDB_INIT_K.min(self.entry.num_rows)
        } else {
            (self.last_k * 2).min(self.entry.num_rows)
        };
        if new_k == self.last_k {
            self.exhausted = true;
            return;
        }
        let result = self
            .entry
            .index
            .lock()
            .unwrap()
            .search(self.query, new_k)
            .unwrap_or_else(|e| panic!("FAISS search failed (k={}): {}", new_k, e));
        for &label in result.labels.iter() {
            if label.is_none() {
                continue;
            }
            let local = label.to_native();
            if local < 0 || (local as usize) >= self.entry.num_rows {
                continue;
            }
            let global_idx = self.entry.node_idx_offset + local as i32;
            if global_idx == self.target_node_idx {
                continue;
            }

            if !self.yielded.insert(global_idx) {
                continue;
            }

            if let Some(target_ts) = self.target_ts {
                let node = get_node(dataset, global_idx);
                if let Some(ts) = node.timestamp.as_ref()
                    && i32::from(*ts) > target_ts
                {
                    continue;
                }
            }
            self.buffer.push(global_idx);
        }
        self.last_k = new_k;
        if new_k >= self.entry.num_rows {
            self.exhausted = true;
        }
    }
}

const PREFER_LATEST_OVERSAMPLE: usize = 4;

fn temporal_bound(target_ts: Option<i32>, cutoff: Option<i32>) -> Option<i32> {
    match (target_ts, cutoff) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

fn edge_same_horizon_task(edge: &ArchivedEdge, target_ts: Option<i32>) -> bool {
    edge.table_type != ArchivedTableType::Db
        && match (edge.timestamp.as_ref(), target_ts) {
            (Some(ts), Some(t)) => i32::from(*ts) == t,
            _ => false,
        }
}

fn edge_visible(edge: &ArchivedEdge, bound: Option<i32>, target_ts: Option<i32>) -> bool {
    !edge_past_bound(edge, bound) && !edge_same_horizon_task(edge, target_ts)
}

fn same_horizon_task_row(node: &ArchivedNode, target: &ArchivedNode) -> bool {
    node.is_task_node && target.timestamp.is_some() && node.timestamp == target.timestamp
}

fn past_bound(node: &ArchivedNode, bound: Option<i32>) -> bool {
    match (node.timestamp.as_ref(), bound) {
        (Some(ts), Some(b)) => i32::from(*ts) > b,
        _ => false,
    }
}

fn edge_past_bound(edge: &ArchivedEdge, bound: Option<i32>) -> bool {
    match (edge.timestamp.as_ref(), bound) {
        (Some(ts), Some(b)) => i32::from(*ts) > b,
        _ => false,
    }
}

fn seed_label_missing(node: &ArchivedNode, target_column: i32) -> bool {
    let cell_i = match node
        .col_name_idxs
        .iter()
        .position(|&c| i32::from(c) == target_column)
    {
        Some(i) => i,
        None => return true,
    };
    match &node.sem_types[cell_i] {
        ArchivedSemType::Number => f32::from(node.number_values[cell_i]).is_nan(),
        ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]).is_nan(),
        ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]).is_nan(),
        ArchivedSemType::Text => false,
    }
}

fn make_pb(len: u64, msg: &'static str, visible: bool) -> ProgressBar {
    if !visible {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg}: {percent:>3}%|\
                 {bar:10}| {pos}/{len} \
                 [{elapsed}<{eta}, {per_sec}]",
            )
            .unwrap()
            .progress_chars("██ "),
    );
    pb.set_message(msg);
    pb
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];

    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(long)]
    pre_dir: String,
    #[arg(default_value = "128")]
    bs: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let sampler = Sampler::new_impl(
        vec![(cli.db_name.clone(), String::new(), 0, 100000)],
        0,
        0,
        1,
        vec![128],
        vec![16],
        0,
        10,
        vec![false],
        0.5,
        "all-MiniLM-L12-v2",
        cli.pre_dir.clone(),
        384,
        0,
        0,
        vec![0_i32],
        vec![Vec::<i32>::new()],
        vec![None],
        -1,
        false,
        false,
        0,
        true,
        false,
        1.0,
        None,
    );
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len(cli.bs));
        let _vecs = sampler.batch(Some(batch_idx), 0, cli.bs, cli.seq_len);
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
