//! Wave64 compute shader emulation and uniformity analysis.

use super::*;

/// Finds whole-wave sections and the values they leave behind.
pub(super) fn compute_sections(p: &Program) -> (Sections, HashSet<Value>) {
    let mut sections = HashMap::new();
    let mut section_out = HashSet::new();
    let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
    for b in 0..p.blocks.len() {
        let found = crate::emit::wave::sections(p, b);
        for (&start, sec) in &found {
            for i in start + 1..sec.end {
                for &(_, w) in &p.blocks[b].insts[i].writes {
                    inside.insert(w, (b, start, sec.end));
                }
            }
        }
        if !found.is_empty() {
            sections.insert(b, found);
        }
    }
    if !inside.is_empty() {
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                section_out.extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        section_out.insert(r);
                    }
                }
            }
        }
    }
    (sections, section_out)
}

/// Whether instruction `(b, i)` lies inside a recognized whole-wave section
/// (`start + 1 .. sec.end`), where `Lower::run` routes it through `section_inst`
/// instead of `inst`.
pub(super) fn in_sections(sections: &Sections, b: usize, i: usize) -> bool {
    sections
        .get(&b)
        .is_some_and(|secs| secs.iter().any(|(&start, sec)| i > start && i < sec.end))
}

/// Immediate post-dominator per block (or virtual exit `n`) using Cooper-Harvey-Kennedy.
/// Unreachable or never-exiting blocks have `rev_idx == usize::MAX`.
pub(super) fn compute_postdoms(n: usize, succs: &[Vec<usize>], fwd: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut rev_succs = vec![Vec::new(); n + 1];
    for b in 0..n {
        if !fwd[b] {
            continue;
        }
        for &s in &succs[b] {
            rev_succs[s].push(b);
        }
    }
    let mut post = Vec::with_capacity(n + 1);
    let mut seen = vec![false; n + 1];
    let mut stack = vec![(n, 0usize)];
    seen[n] = true;
    while let Some((node, idx)) = stack.last_mut() {
        let u = *node;
        if *idx < rev_succs[u].len() {
            let v = rev_succs[u][*idx];
            *idx += 1;
            if !seen[v] {
                seen[v] = true;
                stack.push((v, 0));
            }
        } else {
            post.push(u);
            stack.pop();
        }
    }
    post.reverse();
    let mut rev_idx = vec![usize::MAX; n + 1];
    for (idx, &b) in post.iter().enumerate() {
        rev_idx[b] = idx;
    }
    let mut ipdom = vec![usize::MAX; n + 1];
    ipdom[n] = n;
    let intersect = |mut a: usize, mut b: usize, ipdom: &[usize]| -> usize {
        while a != b {
            while rev_idx[a] > rev_idx[b] {
                a = ipdom[a];
            }
            while rev_idx[b] > rev_idx[a] {
                b = ipdom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &post[1..] {
            let mut new = usize::MAX;
            for &s in &succs[b] {
                if ipdom[s] != usize::MAX {
                    new = if new == usize::MAX {
                        s
                    } else {
                        intersect(s, new, &ipdom)
                    };
                }
            }
            if new != usize::MAX && ipdom[b] != new {
                ipdom[b] = new;
                changed = true;
            }
        }
    }
    (rev_idx, ipdom)
}

/// Computes which IR blocks are thread-group uniform (all active threads arrive
/// together) and which IR values hold group-uniform scalars.
pub(in crate::emit) fn compute_uniformity(
    p: &Program,
    st: &Structured,
    res: &Resources,
    io_inputs: &[(Value, Input)],
) -> (Vec<bool>, Vec<bool>) {
    let n = p.blocks.len();
    let max_v = p
        .inputs
        .iter()
        .map(|i| i.1.0)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis.iter().map(|phi| phi.value.0).chain(
                b.insts
                    .iter()
                    .flat_map(|inst| inst.writes.iter().map(|w| w.1.0)),
            )
        }))
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut val_uniform = vec![true; max_v];
    let io_map: HashMap<Value, Input> = io_inputs.iter().copied().collect();
    let param_set: HashSet<Value> = res.params.iter().map(|&(v, _, _)| v).collect();
    for &(_, v) in &p.inputs {
        let ok = param_set.contains(&v)
            || matches!(io_map.get(&v), Some(Input::GroupId(_) | Input::MeshGroup));
        if !ok {
            val_uniform[v.0 as usize] = false;
        }
    }
    for &k in st.masks.keys().chain(st.highs.keys()) {
        if let Some(u) = val_uniform.get_mut(k.0 as usize) {
            *u = false;
        }
    }

    if n == 0 {
        return (Vec::new(), val_uniform);
    }
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut fwd = vec![false; n];
    for &b in &order {
        fwd[b] = true;
    }
    let mut succs = vec![Vec::new(); n];
    for &b in &order {
        match st.exits[b] {
            LaneExit::Jump(t) => succs[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succs[b].push(taken);
                if not_taken != taken {
                    succs[b].push(not_taken);
                }
            }
            LaneExit::End => succs[b].push(n),
            LaneExit::Indirect => {}
        }
    }
    let (rev_idx, ipdom) = compute_postdoms(n, &succs, &fwd);
    let can_exit: Vec<bool> = (0..n).map(|b| fwd[b] && rev_idx[b] != usize::MAX).collect();
    let mut blocks = can_exit.clone();

    loop {
        let mut changed = false;
        for &b in &order {
            let blk_ok = blocks[b];
            for phi in &p.blocks[b].phis {
                let idx = phi.value.0 as usize;
                if !val_uniform[idx] {
                    continue;
                }
                let preds_ok = blk_ok
                    && p.blocks[b]
                        .preds
                        .iter()
                        .filter(|&&q| fwd[q])
                        .all(|&q| blocks[q]);
                let args_ok = phi.args.iter().all(|a| val_uniform[a.0 as usize]);
                if !(preds_ok && args_ok) {
                    val_uniform[idx] = false;
                    changed = true;
                }
            }
            for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
                let iname = inst.name();
                let uniform_copy = iname == "v_mov_b32"
                    && st.guards[b][ii].is_none()
                    && matches!(
                        &inst.fields,
                        Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none()
                    );
                let scalar_op = matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. })
                    && !iname.starts_with("s_getreg")
                    && !iname.starts_with("s_bcnt")
                    && !iname.starts_with("s_ff1")
                    && !iname.contains("saveexec");
                let reads_ok = blk_ok
                    && if uniform_copy {
                        let Fields::Valu(v) = &inst.fields else {
                            unreachable!()
                        };
                        v.src.first().is_some_and(|op| match *op {
                            Operand::Const(_) | Operand::Literal(_) => true,
                            Operand::Scalar { code, .. } if code != NULL => inst
                                .read(Reg::S(code))
                                .is_some_and(|sv| val_uniform[sv.0 as usize]),
                            Operand::Vector { reg, .. } => inst
                                .read(Reg::V(reg))
                                .is_some_and(|vv| val_uniform[vv.0 as usize]),
                            _ => false,
                        })
                    } else {
                        scalar_op && inst.reads.iter().all(|&(_, v)| val_uniform[v.0 as usize])
                    };
                for &(_, w) in &inst.writes {
                    if val_uniform[w.0 as usize] && !reads_ok {
                        val_uniform[w.0 as usize] = false;
                        changed = true;
                    }
                }
            }
        }
        let mut next_blocks = can_exit.clone();
        for &b in &order {
            if !can_exit[b] {
                continue;
            }
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = st.exits[b]
            {
                let branch_uniform = blocks[b]
                    && st.bools.support(when).iter().all(|a| match *a {
                        Atom::Uniform(v) => val_uniform[v.0 as usize],
                        _ => false,
                    });
                if !branch_uniform {
                    let stop = ipdom[b];
                    let mut stack = vec![taken, not_taken];
                    while let Some(x) = stack.pop() {
                        if x >= n || x == stop || !next_blocks[x] {
                            continue;
                        }
                        next_blocks[x] = false;
                        for &s in &succs[x] {
                            if s < n && s != stop && next_blocks[s] {
                                stack.push(s);
                            }
                        }
                    }
                }
            }
        }
        if next_blocks != blocks {
            blocks = next_blocks;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    (blocks, val_uniform)
}

/// Group-uniform checkpoints that order the LDS accesses of a split console
/// wave. A divergent region cannot contain two accesses without such a
/// checkpoint: placing a group barrier inside it could deadlock.
#[derive(Default)]
pub(in crate::emit) struct LdsFences {
    pub(in crate::emit) enter: HashSet<usize>,
    pub(in crate::emit) leave: HashSet<usize>,
}

pub(in crate::emit) fn plan_lds_fences(p: &Program, uniform: &[bool]) -> Result<LdsFences, ShaderError> {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let memory = |i: &Inst| matches!(i.fields, Fields::Ds { .. }) && i.name() != "ds_swizzle_b32";
    let is_write = |i: &Inst| !i.name().starts_with("ds_read");
    let has_memory: Vec<bool> = p
        .blocks
        .iter()
        .map(|b| b.insts.iter().any(memory))
        .collect();
    let has_read: Vec<bool> = p
        .blocks
        .iter()
        .map(|b| b.insts.iter().any(|i| memory(i) && !is_write(i)))
        .collect();
    let has_write: Vec<bool> = p
        .blocks
        .iter()
        .map(|b| b.insts.iter().any(|i| memory(i) && is_write(i)))
        .collect();
    let fail = |inst: &Inst| ShaderError::Unsupported {
        offset: inst.machine.offset,
        what: "LDS accesses without a uniform LDS fence in split wave64",
    };
    // The two host halves can take different arms at the same time. Multiple pure
    // reads cannot race with each other, whereas any write/atomic conflicts with
    // another read or write anywhere in the same unfenced divergent region.
    for &root in &order {
        if !uniform[root] && root != order[0] {
            continue;
        }
        let mut stack = if uniform[root] {
            p.blocks[root].exit.successors()
        } else {
            vec![root]
        };
        let mut seen = vec![false; p.blocks.len()];
        let mut saw_read = false;
        let mut saw_write = false;
        while let Some(b) = stack.pop() {
            if uniform[b] || seen[b] {
                continue;
            }
            seen[b] = true;
            for inst in p.blocks[b].insts.iter().filter(|i| memory(i)) {
                if is_write(inst) {
                    if saw_write || saw_read {
                        return Err(fail(inst));
                    }
                    saw_write = true;
                } else {
                    if saw_write {
                        return Err(fail(inst));
                    }
                    saw_read = true;
                }
            }
            stack.extend(p.blocks[b].exit.successors());
        }
    }
    // An access not yet ordered by a checkpoint on some incoming path. A
    // uniform block orders incoming accesses at entry and its own at exit.
    let mut pending_read = vec![false; p.blocks.len()];
    let mut pending_write = vec![false; p.blocks.len()];
    loop {
        let mut changed = false;
        for &b in &order {
            if !uniform[b] {
                let next_r =
                    has_read[b] || p.blocks[b].preds.iter().any(|&q| pending_read[q]);
                let next_w =
                    has_write[b] || p.blocks[b].preds.iter().any(|&q| pending_write[q]);
                if next_r && !pending_read[b] {
                    pending_read[b] = true;
                    changed = true;
                }
                if next_w && !pending_write[b] {
                    pending_write[b] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut fences = LdsFences::default();
    for &b in &order {
        let in_read = p.blocks[b].preds.iter().any(|&q| pending_read[q]);
        let in_write = p.blocks[b].preds.iter().any(|&q| pending_write[q]);
        if uniform[b] {
            if in_read || in_write {
                fences.enter.insert(b);
            }
            if has_memory[b] {
                fences.leave.insert(b);
            }
        } else {
            let mut prev_read = in_read;
            let mut prev_write = in_write;
            for inst in p.blocks[b].insts.iter().filter(|i| memory(i)) {
                if is_write(inst) {
                    if prev_write || prev_read {
                        return Err(fail(inst));
                    }
                    prev_write = true;
                } else {
                    if prev_write {
                        return Err(fail(inst));
                    }
                    prev_read = true;
                }
            }
        }
    }
    Ok(fences)
}

/// Masks read as data in a block other than their defining block, keyed by defining block (`None` = entry).
pub(super) fn collect_cross_block_masks(
    p: &Program,
    st: &Structured,
    def_block: &HashMap<Value, usize>,
) -> HashMap<Option<usize>, Vec<Value>> {
    let mut out: HashMap<Option<usize>, Vec<Value>> = HashMap::new();
    let mut record = |v: Value, reader_b: usize| {
        let lo = st.highs.get(&v).copied().unwrap_or(v);
        if !st.masks.contains_key(&lo) {
            return;
        }
        let def_b = def_block.get(&lo).copied();
        if def_b != Some(reader_b) {
            let list = out.entry(def_b).or_default();
            if !list.contains(&lo) {
                list.push(lo);
            }
        }
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            if reads_mask_as_data(st, inst) {
                for &(_, r) in &inst.reads {
                    record(r, b);
                }
            }
        }
        let exit_when = match st.exits[b] {
            LaneExit::Branch { when, .. } => Some(when),
            _ => None,
        };
        for g in st.guards[b].iter().flatten().copied().chain(exit_when) {
            for a in st.bools.support(g) {
                if let Atom::Opaque(v) = a {
                    record(v, b);
                }
            }
        }
    }
    out
}

pub(super) fn operand_const_lane(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

impl<'a> Lower<'a> {
    /// Whether this shader emulates 64-lane waves across multiple host waves or arbitrary width.
    pub(super) fn emulating_wave64(&self) -> bool {
        self.wave64 == Wave64Mode::Emulate
            || (self.wave64 == Wave64Mode::Wave32 && self.total_threads > 32)
    }

    /// Thread-group barrier fencing shared memory only (`barrier(1 | 8)`).
    pub(super) fn lds_barrier(&mut self, block: usize) {
        debug_assert_eq!(self.guard_depth, 0);
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(block, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Ensures `(flat_tid, console_lane, vwave)` exist in `self.s.entry` and returns them.
    pub(super) fn ensure_wave_ids(&mut self) -> (V, V, V) {
        if let Some(ids) = self.wave_ids {
            return ids;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat_in = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let flat = if self.wave64 == Wave64Mode::Emulate {
            flat_in
        } else {
            let first = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[flat_in],
            );
            let wave = self.s.wave_size.expect("set with console lanes");
            let mask = self.s.m.i32(!(wave - 1));
            let base = self.s.m.bin(e, Bin::And, first, mask);
            let lane = self
                .s
                .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            self.s.flags |= flags::WAVE_OPS;
            self.s.m.bin(e, Bin::Add, base, lane)
        };
        let k63 = self.s.m.i32(63);
        let k6 = self.s.m.i32(6);
        let console_lane = self.s.m.bin(e, Bin::And, flat, k63);
        let vwave = self.s.m.bin(e, Bin::LShr, flat, k6);
        let ids = (flat, console_lane, vwave);
        self.wave_ids = Some(ids);
        ids
    }

    /// Pointer to LDS scratch dword `base + vwave * stride + offset`.
    pub(super) fn scratch_ptr(&mut self, block: usize, base: u32, stride: u32, vwave: V, offset: u32) -> V {
        let base_v = self.s.m.i32(base);
        let stride_v = self.s.m.i32(stride);
        let wave_off = self.s.m.bin(block, Bin::Mul, vwave, stride_v);
        let mut idx = self.s.m.bin(block, Bin::Add, base_v, wave_off);
        if offset != 0 {
            let off_v = self.s.m.i32(offset);
            idx = self.s.m.bin(block, Bin::Add, idx, off_v);
        }
        let lds = self.s.lds();
        self.s.m.element(block, lds, idx)
    }

    /// Pointer to LDS exchange dword `exchange_base + vwave * 64 + lane_offset`.
    pub(super) fn exchange_ptr(&mut self, block: usize, vwave: V, lane_offset: V) -> V {
        let base_v = self.s.m.i32(self.exchange_base);
        let k64 = self.s.m.i32(64);
        let wave_off = self.s.m.bin(block, Bin::Mul, vwave, k64);
        let row = self.s.m.bin(block, Bin::Add, base_v, wave_off);
        let idx = self.s.m.bin(block, Bin::Add, row, lane_offset);
        let lds = self.s.lds();
        self.s.m.element(block, lds, idx)
    }

    /// Applies associative `op` to two `i32` bit-patterns in the current block.
    pub(super) fn apply_wave_op(&mut self, op: crate::emit::wave::Op, a: V, b: V) -> V {
        use crate::emit::wave::Op;
        let c = self.cur;
        match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, a, b),
            Op::Or => self.s.m.bin(c, Bin::Or, a, b),
            Op::And => self.s.m.bin(c, Bin::And, a, b),
            Op::Xor => self.s.m.bin(c, Bin::Xor, a, b),
            Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                let p = match op {
                    Op::MinU => pred::IULT,
                    Op::MaxU => pred::IUGT,
                    Op::MinI => pred::SLT,
                    _ => pred::SGT,
                };
                let cmp = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, cmp, a, b)
            }
            Op::AddF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.s.m.bin(c, Bin::Add, fa, fb);
                self.as_int(r)
            }
            Op::MaxF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.binary(35, fa, fb);
                self.as_int(r)
            }
            Op::MinF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.binary(36, fa, fb);
                self.as_int(r)
            }
            Op::AddH => {
                let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
                let fa = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[a]);
                let fb = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[b]);
                let sum = self.s.m.bin(c, Bin::Add, fa, fb);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[sum])
            }
        }
    }

    /// Exchanges one `i32` per 32-lane half across each virtual 64-lane wave using `ballot_base`.
    /// Must run in `self.cur` in convergent control flow (`uniform_here = true`) outside any guard.
    pub(super) fn exchange_two_halves(
        &mut self,
        val: V,
        partial_h1_default: V,
        uniform_here: bool,
    ) -> Result<(V, V), Why> {
        if self.guard_depth > 0 || !uniform_here {
            return Err("a cross-lane wave operation in divergent control flow");
        }
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        let stride = self.ballot_stride;
        let base = self.ballot_base;
        self.lds_barrier(self.cur);
        let k0 = self.s.m.i32(0);
        let k32 = self.s.m.i32(32);
        let is_h0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
        let at0 = self.begin_guarded(Some(is_h0));
        let p0 = self.scratch_ptr(self.cur, base, stride, vwave, 0);
        self.s.m.store(self.cur, p0, val);
        let tail = self.total_threads & 63;
        if tail != 0 && tail <= 32 {
            let last_w = self.s.m.i32(self.virtual_waves - 1);
            let is_last = self.s.m.cmp(self.cur, pred::EQ, vwave, last_w);
            let at_last = self.begin_guarded(Some(is_last));
            let p1 = self.scratch_ptr(self.cur, base, stride, vwave, 1);
            self.s.m.store(self.cur, p1, partial_h1_default);
            self.end_guarded(at_last, &[]);
        }
        self.end_guarded(at0, &[]);

        let is_h1 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k32);
        let at1 = self.begin_guarded(Some(is_h1));
        let p1 = self.scratch_ptr(self.cur, base, stride, vwave, 1);
        self.s.m.store(self.cur, p1, val);
        self.end_guarded(at1, &[]);

        self.lds_barrier(self.cur);
        let c = self.cur;
        let r0_ptr = self.scratch_ptr(c, base, stride, vwave, 0);
        let r1_ptr = self.scratch_ptr(c, base, stride, vwave, 1);
        Ok((self.s.m.load(c, r0_ptr), self.s.m.load(c, r1_ptr)))
    }

    /// Exchanges `val` from every console lane across the virtual 64-lane wave via LDS
    /// and returns the value held by `lane`. Requires a uniform block.
    pub(super) fn exchange_64(&mut self, val: V, lane: V) -> Result<V, Why> {
        if self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false) {
            return Err("a cross-lane read in divergent control flow");
        }
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        self.lds_barrier(self.cur);
        let wptr = self.exchange_ptr(self.cur, vwave, console_lane);
        self.s.m.store(self.cur, wptr, val);
        self.lds_barrier(self.cur);
        let k63 = self.s.m.i32(63);
        let idx = self.s.m.bin(self.cur, Bin::And, lane, k63);
        let rptr = self.exchange_ptr(self.cur, vwave, idx);
        Ok(self.s.m.load(self.cur, rptr))
    }

    /// A read at the start of the active EXEC edge can elect across the whole
    /// guest wave while all host waves are still together in its predecessor.
    pub(super) fn elect_before_exec_branch(&mut self, b: usize) -> Result<(), Stop> {
        let block = &self.x.p.blocks[b];
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.x.st.exits[b]
        else {
            return Ok(());
        };
        let branch = block.insts.last().expect("branch instruction");
        let (active, on_taken) = match branch.name() {
            "s_cbranch_execz" => (not_taken, false),
            "s_cbranch_execnz" => (taken, true),
            _ => return Ok(()),
        };
        let next = &self.x.p.blocks[active];
        if !self.uniform[b]
            || self.uniform[active]
            || next.preds.as_slice() != [b]
            || self.rpo[active] <= self.rpo[b]
        {
            return Ok(());
        }
        let Some(read) = next.insts.first() else {
            return Ok(());
        };
        if read.name() != "v_readfirstlane_b32"
            || read
                .writes
                .first()
                .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1))
        {
            return Ok(());
        }
        let Fields::Valu(v) = &read.fields else {
            unreachable!()
        };
        let [Operand::Vector { reg, .. }] = v.src[..] else {
            return Ok(());
        };
        let Some(src) = read.read(Reg::V(reg)) else {
            return Ok(());
        };
        // A successor phi is unavailable here. A DPP source might itself
        // read another lane, so leave that form to the ordinary lowering.
        if !self.vals.contains_key(&src) || self.val_uniform[src.0 as usize] || v.dpp.is_some() {
            return Ok(());
        }
        let at = read.machine.offset;
        let Ok(value) = self.src(read, 0) else {
            return Ok(());
        };
        let condition = self
            .boolean(when, self.cur)
            .map_err(|w| unsupported(at, w))?;
        let enabled = if on_taken {
            condition
        } else {
            let yes = self.s.m.bool(true);
            self.s.m.bin(self.cur, Bin::Xor, condition, yes)
        };
        let picked = self
            .emulated_readfirstlane(value, Some(enabled))
            .map_err(|w| unsupported(at, w))?;
        self.firstlane_at_entry.insert(active, picked);
        Ok(())
    }

    /// Reads `val` from the first active lane of the host hardware wave.
    pub(super) fn host_wave_readfirstlane(&mut self, val: V, act: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
        self.s.flags |= flags::WAVE_OPS;
        let lo = self.s.m.extract(c, ballot, 0);
        let hi = self.s.m.extract(c, ballot, 1);
        let k0 = self.s.m.i32(0);
        let k32 = self.s.m.i32(32);
        let bit_lo = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let bit_hi = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let hi_idx = self.s.m.bin(c, Bin::Add, bit_hi, k32);
        let has_hi = self.s.m.cmp(c, pred::NE, hi, k0);
        let idx = self.s.m.select(c, has_hi, hi_idx, k0);
        let has_lo = self.s.m.cmp(c, pred::NE, lo, k0);
        let idx = self.s.m.select(c, has_lo, bit_lo, idx);
        self.read_lane(val, idx)
    }

    /// Emulates `v_readfirstlane_b32(val)` across all 64 lanes of a virtual wave.
    pub(super) fn emulated_readfirstlane(&mut self, val: V, guard: Option<V>) -> Result<V, Why> {
        let divergent =
            self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false);
        let act = guard.unwrap_or_else(|| self.s.m.bool(true));
        if divergent && self.wave64 != Wave64Mode::Wave32 {
            // Inside divergent control flow the virtual wave halves cannot meet at a
            // barrier; read the first active lane of the host wave instead.
            return Ok(self.host_wave_readfirstlane(val, act));
        }
        if self.wave64 == Wave64Mode::Wave32 {
            let c = self.cur;
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let r = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
            self.s.flags |= flags::WAVE_OPS;
            let mask = self.s.m.extract(c, r, 0);
            let k0 = self.s.m.i32(0);
            let any = self.s.m.cmp(c, pred::NE, mask, k0);
            let bit = self
                .s
                .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[mask]);
            let safe_idx = self.s.m.select(c, any, bit, k0);
            let picked = self.read_lane(val, safe_idx);
            if divergent {
                // shortcut: in divergent control flow the halves cannot meet at a
                // barrier, so each 32-lane half reads its own first active lane.
                // Exact when the value is uniform across active lanes or inside a
                // waterfall loop visiting distinct values half by half.
                return Ok(picked);
            }
            let (_, console_lane, vwave) = self.ensure_wave_ids();
            let stride = self.ballot_stride;
            let base = self.ballot_base;

            self.lds_barrier(self.cur);
            let k0 = self.s.m.i32(0);
            let k32 = self.s.m.i32(32);
            let is_h0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
            let at0 = self.begin_guarded(Some(is_h0));
            let pm0 = self.scratch_ptr(self.cur, base, stride, vwave, 0);
            let pv0 = self.scratch_ptr(self.cur, base, stride, vwave, 1);
            self.s.m.store(self.cur, pm0, mask);
            self.s.m.store(self.cur, pv0, picked);
            let tail = self.total_threads & 63;
            if tail != 0 && tail <= 32 {
                let last_w = self.s.m.i32(self.virtual_waves - 1);
                let is_last = self.s.m.cmp(self.cur, pred::EQ, vwave, last_w);
                let at_last = self.begin_guarded(Some(is_last));
                let pm1 = self.scratch_ptr(self.cur, base, stride, vwave, 2);
                self.s.m.store(self.cur, pm1, k0);
                self.end_guarded(at_last, &[]);
            }
            self.end_guarded(at0, &[]);

            let is_h1 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k32);
            let at1 = self.begin_guarded(Some(is_h1));
            let pm1 = self.scratch_ptr(self.cur, base, stride, vwave, 2);
            let pv1 = self.scratch_ptr(self.cur, base, stride, vwave, 3);
            self.s.m.store(self.cur, pm1, mask);
            self.s.m.store(self.cur, pv1, picked);
            self.end_guarded(at1, &[]);

            self.lds_barrier(self.cur);
            let c = self.cur;
            let pm0 = self.scratch_ptr(c, base, stride, vwave, 0);
            let pv0 = self.scratch_ptr(c, base, stride, vwave, 1);
            let pm1 = self.scratch_ptr(c, base, stride, vwave, 2);
            let pv1 = self.scratch_ptr(c, base, stride, vwave, 3);
            let m0 = self.s.m.load(c, pm0);
            let m1 = self.s.m.load(c, pm1);
            let v0 = self.s.m.load(c, pv0);
            let v1 = self.s.m.load(c, pv1);
            // GFX10 `v_readfirstlane_b32` with `EXEC == 0` returns lane 0 (`v0`).
            let has0 = self.s.m.cmp(c, pred::NE, m0, k0);
            let no1 = self.s.m.cmp(c, pred::EQ, m1, k0);
            let use0 = self.s.m.bin(c, Bin::Or, has0, no1);
            return Ok(self.s.m.select(c, use0, v0, v1));
        }

        // Emulate mode: find lowest active console lane via atomicMin.
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        let stride = self.ballot_stride;
        let base = self.ballot_base;
        self.lds_barrier(self.cur);
        let k0 = self.s.m.i32(0);
        let kmax = self.s.m.i32(u32::MAX);
        let is_init = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
        let at0 = self.begin_guarded(Some(is_init));
        let pw = self.scratch_ptr(self.cur, base, stride, vwave, 0);
        let pv = self.scratch_ptr(self.cur, base, stride, vwave, 1);
        self.s.m.store(self.cur, pw, kmax);
        self.s.m.store(self.cur, pv, k0);
        self.end_guarded(at0, &[]);

        self.lds_barrier(self.cur);
        let at_act = self.begin_guarded(guard);
        let pw = self.scratch_ptr(self.cur, base, stride, vwave, 0);
        let _ = self
            .s
            .m
            .atomic(self.cur, crate::emit::bitcode::Rmw::UMin, pw, console_lane);
        self.end_guarded(at_act, &[]);

        self.lds_barrier(self.cur);
        let pw = self.scratch_ptr(self.cur, base, stride, vwave, 0);
        let winner = self.s.m.load(self.cur, pw);
        let has_win = self.s.m.cmp(self.cur, pred::NE, winner, kmax);
        let fallback_lane = self.s.m.select(self.cur, has_win, winner, k0);
        let is_win = self
            .s
            .m
            .cmp(self.cur, pred::EQ, fallback_lane, console_lane);
        let at_win = self.begin_guarded(Some(is_win));
        let pv = self.scratch_ptr(self.cur, base, stride, vwave, 1);
        self.s.m.store(self.cur, pv, val);
        self.end_guarded(at_win, &[]);

        self.lds_barrier(self.cur);
        let pv = self.scratch_ptr(self.cur, base, stride, vwave, 1);
        Ok(self.s.m.load(self.cur, pv))
    }

    /// Pre-emits cross-block wave ballots while `self.cur` is at the end of `def_b` (`None` = entry).
    pub(super) fn preemit_masks(&mut self, def_b: Option<usize>) {
        let Some(los) = self.cross_block_masks.get(&def_b).cloned() else {
            return;
        };
        let uniform_here = match def_b {
            None => true,
            Some(b) => self.uniform.get(b).copied().unwrap_or(false),
        };
        for lo in los {
            if self.ballots.contains_key(&lo) || self.ballot_fail.contains_key(&lo) {
                continue;
            }
            if let Err(why) = self.compute_ballot_in_cur(lo, uniform_here) {
                self.ballot_fail.insert(lo, why);
            }
        }
    }

    /// Materializes 64-lane ballots needed as data or opaque guards by `(b, i)` before `inst` runs.
    pub(super) fn materialize_masks(&mut self, b: usize, i: usize, inst: &Inst) {
        let mut todo = Vec::new();
        let mut add = |v: Value| {
            let lo = self.x.st.highs.get(&v).copied().unwrap_or(v);
            if self.x.st.masks.contains_key(&lo)
                && !self.ballots.contains_key(&lo)
                && !self.ballot_fail.contains_key(&lo)
                && !todo.contains(&lo)
            {
                todo.push(lo);
            }
        };
        if self.mask_ok && reads_mask_as_data(self.x.st, inst) {
            for &(_, r) in &inst.reads {
                add(r);
            }
        }
        if let Some(g) = self.x.st.guards[b][i] {
            for a in self.x.st.bools.support(g) {
                if let Atom::Opaque(v) = a {
                    add(v);
                }
            }
        }
        let uniform_here = self.uniform.get(b).copied().unwrap_or(false);
        for lo in todo {
            if let Err(why) = self.compute_ballot_in_cur(lo, uniform_here) {
                self.ballot_fail.insert(lo, why);
            }
        }
    }

    /// Emits a 64-lane ballot for boolean `bit` into `self.cur`.
    pub(super) fn compute_ballot_bit(&mut self, bit: V, uniform_here: bool) -> Result<(V, V), Why> {
        let c = self.cur;
        if self.wave64 == Wave64Mode::Emulate {
            if self.guard_depth > 0 || !uniform_here {
                return Err("a 64-lane wave ballot in divergent control flow");
            }
            let (_, console_lane, vwave) = self.ensure_wave_ids();
            let base = self.ballot_base;
            let stride = self.ballot_stride;
            self.lds_barrier(self.cur);
            let k0 = self.s.m.i32(0);
            let is0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
            let at0 = self.begin_guarded(Some(is0));
            let p0 = self.scratch_ptr(self.cur, base, stride, vwave, 0);
            let p1 = self.scratch_ptr(self.cur, base, stride, vwave, 1);
            self.s.m.store(self.cur, p0, k0);
            self.s.m.store(self.cur, p1, k0);
            self.end_guarded(at0, &[]);

            self.lds_barrier(self.cur);
            let at_bit = self.begin_guarded(Some(bit));
            let bcur = self.cur;
            let k32 = self.s.m.i32(32);
            let k31 = self.s.m.i32(31);
            let k1 = self.s.m.i32(1);
            let half = self.s.m.cmp(bcur, pred::IUGE, console_lane, k32);
            let half_off = self.s.m.cast(bcur, Cast::ZExt, half, self.s.t.i32);
            let base_v = self.s.m.i32(base);
            let stride_v = self.s.m.i32(stride);
            let wave_off = self.s.m.bin(bcur, Bin::Mul, vwave, stride_v);
            let slot_base = self.s.m.bin(bcur, Bin::Add, base_v, wave_off);
            let slot_idx = self.s.m.bin(bcur, Bin::Add, slot_base, half_off);
            let lds = self.s.lds();
            let ptr = self.s.m.element(bcur, lds, slot_idx);
            let sh = self.s.m.bin(bcur, Bin::And, console_lane, k31);
            let mask_bit = self.s.m.bin(bcur, Bin::Shl, k1, sh);
            let _ = self
                .s
                .m
                .atomic(bcur, crate::emit::bitcode::Rmw::Or, ptr, mask_bit);
            self.end_guarded(at_bit, &[]);

            self.lds_barrier(self.cur);
            let c = self.cur;
            let p0 = self.scratch_ptr(c, base, stride, vwave, 0);
            let p1 = self.scratch_ptr(c, base, stride, vwave, 1);
            Ok((self.s.m.load(c, p0), self.s.m.load(c, p1)))
        } else {
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let r = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
            self.s.flags |= flags::WAVE_OPS;
            let half = self.s.m.extract(c, r, 0);
            let zero = self.s.m.i32(0);
            self.exchange_two_halves(half, zero, uniform_here)
        }
    }

    /// Emits a 64-lane ballot for active mask `lo` into `self.cur`.
    pub(super) fn compute_ballot_in_cur(&mut self, lo: Value, uniform_here: bool) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        let xy = if !self.x.st.active.contains(&lo) {
            self.mask_number(lo)?
        } else {
            let c = self.cur;
            if !self.balloting.insert(lo) {
                return Err("a lane mask whose lanes depend on its own value");
            }
            let bit = self.boolean(m, c);
            self.balloting.remove(&lo);
            self.compute_ballot_bit(bit?, uniform_here)?
        };
        self.ballots.insert(lo, xy);
        Ok(xy)
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    pub(super) fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }
}

impl<'a> Lower<'a> {
    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    pub(super) fn shape(&mut self, shape: crate::emit::wave::Shape) -> Result<V, Why> {
        use crate::emit::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
            Shape::Half(_, src, mask, ident, hi) => {
                // The high half is a console lane's: compute only.
                if self.mask_data != MaskData::Console {
                    return Err("a 16-bit wave reduction outside compute");
                }
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                let x = self.s.m.select(c, on, x, id);
                self.s.flags |= flags::WAVE_OPS;
                // shortcut: the f16 sum is taken in f32 and rounded once; the
                // console rounds each step, in its own order.
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
                let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
                let r = self.s.op(
                    c,
                    "waveActiveOp.f32",
                    Attr::NoUnwind,
                    self.s.t.f32,
                    119,
                    &[f, sum, signed],
                );
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let r = self.s.m.bin(c, Bin::And, r, low);
                // Lane `hi` runs the section on the console even where it is
                // elsewhere on the host; there it holds the identity.
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let b = self
                    .s
                    .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
                let word = self.s.m.extract(c, b, hi / 32);
                let (sh, one, z) = (self.s.m.i32(hi % 32), self.s.m.i32(1), self.s.m.i32(0));
                let bit = self.s.m.bin(c, Bin::LShr, word, sh);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                let there = self.s.m.cmp(c, pred::NE, bit, z);
                let lane = self.s.m.i32(hi);
                let at = self.read_lane(x, lane);
                let top = self.s.m.select(c, there, at, id);
                let high = self.s.m.i32(0xFFFF_0000);
                let top = self.s.m.bin(c, Bin::And, top, high);
                return Ok(self.s.m.bin(c, Bin::Or, top, r));
            }
        };
        // A mesh program's lanes are numbered as the console's, its host
        // waves are not the console's: only a minimum or maximum (the loop
        // that serves each distinct value once) means the same on any wave.
        if self.mask_data == MaskData::None
            && (prefix.is_some()
                || !matches!(
                    op,
                    Op::MinU | Op::MaxU | Op::MinI | Op::MaxI | Op::MinF | Op::MaxF
                ))
        {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);

        if self.wave64 == Wave64Mode::Emulate {
            if self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false) {
                return Err("a wave reduction in divergent control flow");
            }
            let (_, console_lane, vwave) = self.ensure_wave_ids();
            let max_in_vwave = self.total_threads.min(64);
            if let Some(inclusive) = prefix {
                if !matches!(op, Op::AddU | Op::AddF) {
                    return Err("a wave prefix of an operation other than a sum");
                }
                let mut cur_val = x;
                for shift in 0..6 {
                    let stride = 1u32 << shift;
                    if stride >= max_in_vwave {
                        break;
                    }
                    self.lds_barrier(self.cur);
                    let wp = self.exchange_ptr(self.cur, vwave, console_lane);
                    self.s.m.store(self.cur, wp, cur_val);
                    self.lds_barrier(self.cur);
                    let ks = self.s.m.i32(stride);
                    let active = self.s.m.cmp(self.cur, pred::IUGE, console_lane, ks);
                    let at = self.begin_guarded(Some(active));
                    let partner_lane = self.s.m.bin(self.cur, Bin::Sub, console_lane, ks);
                    let rp = self.exchange_ptr(self.cur, vwave, partner_lane);
                    let partner = self.s.m.load(self.cur, rp);
                    let sum = self.apply_wave_op(op, partner, cur_val);
                    cur_val = self.end_guarded(at, &[(sum, cur_val)])[0];
                }
                if !inclusive {
                    self.lds_barrier(self.cur);
                    let wp = self.exchange_ptr(self.cur, vwave, console_lane);
                    self.s.m.store(self.cur, wp, cur_val);
                    self.lds_barrier(self.cur);
                    let k0 = self.s.m.i32(0);
                    let k1 = self.s.m.i32(1);
                    let not0 = self.s.m.cmp(self.cur, pred::NE, console_lane, k0);
                    let at = self.begin_guarded(Some(not0));
                    let prev_lane = self.s.m.bin(self.cur, Bin::Sub, console_lane, k1);
                    let rp = self.exchange_ptr(self.cur, vwave, prev_lane);
                    let prev = self.s.m.load(self.cur, rp);
                    cur_val = self.end_guarded(at, &[(prev, id)])[0];
                }
                return Ok(cur_val);
            } else {
                self.lds_barrier(self.cur);
                let wp = self.exchange_ptr(self.cur, vwave, console_lane);
                self.s.m.store(self.cur, wp, x);
                let partial = !self.total_threads.is_multiple_of(64);
                let (flat, _, _) = self.ensure_wave_ids();
                let total_v = self.s.m.i32(self.total_threads);
                for shift in (0..6).rev() {
                    let stride = 1u32 << shift;
                    if stride >= max_in_vwave {
                        continue;
                    }
                    self.lds_barrier(self.cur);
                    let b = self.cur;
                    let ks = self.s.m.i32(stride);
                    let mut active = self.s.m.cmp(b, pred::IULT, console_lane, ks);
                    if partial {
                        let partner_flat = self.s.m.bin(b, Bin::Add, flat, ks);
                        let in_bounds = self.s.m.cmp(b, pred::IULT, partner_flat, total_v);
                        active = self.s.m.bin(b, Bin::And, active, in_bounds);
                    }
                    let at = self.begin_guarded(Some(active));
                    let cb = self.cur;
                    let partner_lane = self.s.m.bin(cb, Bin::Add, console_lane, ks);
                    let pa = self.exchange_ptr(cb, vwave, console_lane);
                    let pb = self.exchange_ptr(cb, vwave, partner_lane);
                    let va = self.s.m.load(cb, pa);
                    let vb = self.s.m.load(cb, pb);
                    let comb = self.apply_wave_op(op, va, vb);
                    let pw = self.exchange_ptr(cb, vwave, console_lane);
                    self.s.m.store(cb, pw, comb);
                    self.end_guarded(at, &[]);
                }
                self.lds_barrier(self.cur);
                let k0 = self.s.m.i32(0);
                let rp = self.exchange_ptr(self.cur, vwave, k0);
                return Ok(self.s.m.load(self.cur, rp));
            }
        }

        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddH, _) => return Err("a wave reduction over 16-bit floats"),
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if self.emulating_wave64() {
                    if self.guard_depth > 0
                        || !self.uniform.get(self.cur_ir).copied().unwrap_or(false)
                    {
                        return Err("a wave prefix in divergent control flow");
                    }
                    let half_total = self.s.m.bin(c, Bin::Add, r, arg);
                    let half_total_bits = if float {
                        self.as_int(half_total)
                    } else {
                        half_total
                    };
                    let local = if inclusive { half_total } else { r };
                    let local_bits = if float { self.as_int(local) } else { local };
                    let (_, console_lane, vwave) = self.ensure_wave_ids();
                    let stride = self.ballot_stride;
                    let base = self.ballot_base;
                    self.lds_barrier(self.cur);
                    let k31 = self.s.m.i32(31);
                    let is31 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k31);
                    let at = self.begin_guarded(Some(is31));
                    let ptr = self.scratch_ptr(self.cur, base, stride, vwave, 0);
                    self.s.m.store(self.cur, ptr, half_total_bits);
                    self.end_guarded(at, &[]);
                    self.lds_barrier(self.cur);
                    let ptr = self.scratch_ptr(self.cur, base, stride, vwave, 0);
                    let h0_bits = self.s.m.load(self.cur, ptr);
                    let combined = self.apply_wave_op(op, h0_bits, local_bits);
                    let k32 = self.s.m.i32(32);
                    let in_h0 = self.s.m.cmp(self.cur, pred::IULT, console_lane, k32);
                    return Ok(self.s.m.select(self.cur, in_h0, local_bits, combined));
                }
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        let local_bits = if float { self.as_int(r) } else { r };
        if self.emulating_wave64() {
            let uniform_here = self.uniform.get(self.cur_ir).copied().unwrap_or(false);
            let (h0, h1) = self.exchange_two_halves(local_bits, id, uniform_here)?;
            return Ok(self.apply_wave_op(op, h0, h1));
        }
        Ok(local_bits)
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    pub(super) fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Whether a quad-local read must reach helper lanes: in a pixel program
    /// every quad a triangle edge crosses has some (the console's whole-quad
    /// mode runs them; discarded pixels stay helpers too), and only quad
    /// reads see them on the host (wave reads give 0 from them on AMD: wrong
    /// derivatives of computed values along every triangle edge).
    pub(super) fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out (DXIL before
    /// `WaveOpsIncludeHelperLanes`): once only helper lanes are left in such
    /// a loop, `WaveReadLaneFirst` reads no lane, no lane matches, and the
    /// loop runs to [`LOOP_LIMIT`] in every such lane: draws of 0.3 to 1.3 s
    /// each, then DEVICE_HUNG. With its own value a helper lane matches in
    /// the first pass and leaves; what a helper lane computes only feeds
    /// derivatives, and its stores are dropped.
    pub(super) fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    pub(super) fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// Whether `src` is active under the current `EXEC` mask (`guard` for
    /// straight-line narrowing, plus the host wave's live lanes for
    /// branch-structurised control flow). Returns `None` for unguarded pixel
    /// quad ops so WQM helper lanes remain active (`waveActiveBallot` excludes
    /// helper lanes in DXIL).
    pub(super) fn src_lane_active(&mut self, src: V, guard: Option<V>, mesh_quad: bool) -> Option<V> {
        let c = self.cur;
        if self.s.pixel {
            let g = guard?;
            let (one, zero) = (self.s.m.i32(1), self.s.m.i32(0));
            let active_i32 = self.s.m.select(c, g, one, zero);
            let picked = self.quad_pick(active_i32, src);
            return Some(self.s.m.cmp(c, pred::NE, picked, zero));
        }
        if self.emulating_wave64() && !mesh_quad {
            let g = guard?;
            let uniform_here = self.uniform.get(self.cur_ir).copied().unwrap_or(false);
            let (lo, hi) = self.compute_ballot_bit(g, uniform_here).ok()?;
            return Some(self.lane_bit_of(c, src, lo, hi));
        }
        let bit = guard.unwrap_or_else(|| self.s.m.bool(true));
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let (lo, hi) = (
            self.s.m.extract(c, ballot, 0),
            self.s.m.extract(c, ballot, 1),
        );
        let ballot_lane = if mesh_quad {
            let host_lane = self
                .s
                .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let base = self.s.m.bin(c, Bin::And, host_lane, not3);
            let q = self.s.m.bin(c, Bin::And, src, three);
            self.s.m.bin(c, Bin::Or, base, q)
        } else {
            src
        };
        Some(self.lane_bit_of(c, ballot_lane, lo, hi))
    }
}


