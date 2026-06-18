import sys
import os
import copy
import json
import math
import random
from random import randint
import traceback
import numpy as np
import gc
import csv
import threading
import queue
import time
from collections import defaultdict, deque

sys.modules["markupsafe._speedups"] = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import schedulefree
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from sim_hand import Hand
from cfr_sim_encoder import CfrSimEncoder


class Profiler:
    def __init__(self):
        self.stats = defaultdict(float)
        self.counts = defaultdict(int)
        self.lock = threading.Lock()

    class Timer:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            self.start = time.perf_counter()

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            with self.profiler.lock:
                self.profiler.stats[self.name] += elapsed
                self.profiler.counts[self.name] += 1

    def profile(self, name):
        return self.Timer(self, name)

    def report_and_reset(self, global_updates):
        with self.lock:
            print(f"\n========== RUNTIME PROFILING REPORT (Update {global_updates}) ==========")
            for name, total_time in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
                calls = self.counts[name]
                avg_ms = (total_time / calls) * 1000 if calls > 0 else 0
                print(f"{name:<40} | {total_time:<15.4f} | {calls:<8} | {avg_ms:<15.4f}")
            print("========================================================================\n")
            self.stats.clear()
            self.counts.clear()


global_profiler = Profiler()


class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.value

    def increment(self):
        with self.lock:
            self.value += 1


class SubgameSolver:
    """Tracks root policy changes to switch dynamically between baseline self-play and deeper subtree exploration."""
    def __init__(self,
                 threshold: float = 0.0001,
                 cooldown_duration: int = 500,
                 kl_ema_decay: float = 0.98,
                 warmup_steps: int = 500,
                 focus_nsims_multiplier: float = 1.5,
                 min_samples_before_trigger: int = 100,
                 focus_schedule=((3, 100), (2, 200), (1, 300))):
        self.threshold = threshold
        self.cooldown_duration = cooldown_duration
        self.kl_ema_decay = kl_ema_decay
        self.warmup_steps = warmup_steps
        self.focus_nsims_multiplier = focus_nsims_multiplier
        self.min_samples_before_trigger = min_samples_before_trigger
        self.focus_schedule = tuple(focus_schedule)
        self.focus_duration = sum(d for _, d in self.focus_schedule)

        self._is_focused = False
        self._focus_remaining = 0
        self._cooldown_remaining = 0
        self._activation_count = 0
        self._steps_observed = 0
        self._schedule_pos = 0
        self._substep_remaining = 0
        self._preflop_kl_ema = 0.0
        self._preflop_kl_samples = 0
        self._last_trigger_step = -1
        self._last_release_step = -1
        self._total_focused_steps = 0
        self._lock = threading.Lock()

    def record_preflop_kl(self, kl_value: float):
        if kl_value is None or not np.isfinite(kl_value):
            return
        with self._lock:
            if self._preflop_kl_samples == 0:
                self._preflop_kl_ema = kl_value
            else:
                d = self.kl_ema_decay
                self._preflop_kl_ema = d * self._preflop_kl_ema + (1.0 - d) * kl_value
            self._preflop_kl_samples += 1

    def maybe_trigger(self, current_step: int):
        with self._lock:
            if self._is_focused or self._cooldown_remaining > 0:
                return False
            if self._steps_observed < self.warmup_steps:
                return False
            if self._preflop_kl_samples < self.min_samples_before_trigger:
                return False
            if self._preflop_kl_ema > self.threshold:
                self._is_focused = True
                self._focus_remaining = self.focus_duration
                self._schedule_pos = 0
                self._substep_remaining = self.focus_schedule[0][1]
                self._activation_count += 1
                self._last_trigger_step = current_step
                self._preflop_kl_ema = 0.0
                self._preflop_kl_samples = 0
                return True
        return False

    def tick(self, current_step: int):
        with self._lock:
            self._steps_observed += 1
            if self._is_focused:
                self._focus_remaining -= 1
                self._substep_remaining -= 1
                self._total_focused_steps += 1

                street_transition = None
                if (self._substep_remaining <= 0
                        and self._schedule_pos < len(self.focus_schedule) - 1):
                    self._schedule_pos += 1
                    self._substep_remaining = self.focus_schedule[self._schedule_pos][1]
                    street_transition = ('focus_street_advanced',
                                          self.focus_schedule[self._schedule_pos][0],
                                          current_step)

                if self._focus_remaining <= 0:
                    self._is_focused = False
                    self._cooldown_remaining = self.cooldown_duration
                    self._last_release_step = current_step
                    self._preflop_kl_ema = 0.0
                    self._preflop_kl_samples = 0
                    return ('released', current_step)
                return street_transition
            elif self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                if self._cooldown_remaining == 0:
                    return ('cooldown_ended', current_step)
        return None

    def current_focus_street(self):
        with self._lock:
            if self._is_focused:
                return self.focus_schedule[self._schedule_pos][0]
            return None

    def in_focus(self) -> bool:
        with self._lock:
            return self._is_focused

    def status(self) -> dict:
        with self._lock:
            if self._is_focused:
                state = 'FOCUSED'
                phase_remaining = self._focus_remaining
                focus_street = self.focus_schedule[self._schedule_pos][0]
            elif self._cooldown_remaining > 0:
                state = 'COOLDOWN'
                phase_remaining = self._cooldown_remaining
                focus_street = None
            else:
                state = 'IDLE'
                phase_remaining = 0
                focus_street = None
            return {
                'state': state,
                'phase_remaining': phase_remaining,
                'focus_street': focus_street,
                'substep_remaining': self._substep_remaining if self._is_focused else 0,
                'preflop_kl_ema': self._preflop_kl_ema,
                'kl_samples': self._preflop_kl_samples,
                'activations': self._activation_count,
                'total_focused_steps': self._total_focused_steps,
                'last_trigger_step': self._last_trigger_step,
                'last_release_step': self._last_release_step,
            }


class RoleBaseline:
    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.values = defaultdict(float)
        self.counts = defaultdict(int)
        self.lock = threading.Lock()

    def get(self, street: int, position: int) -> float:
        with self.lock:
            return self.values[(street, position)]

    def update(self, street: int, position: int, value_bb: float):
        with self.lock:
            key = (street, position)
            if self.counts[key] == 0:
                self.values[key] = value_bb
            else:
                self.values[key] = self.decay * self.values[key] + (1.0 - self.decay) * value_bb
            self.counts[key] += 1

    def snapshot(self):
        with self.lock:
            return {k: (v, self.counts[k]) for k, v in self.values.items()}


class QPivotLogger:
    def __init__(self, capacity_per_street: int = 50):
        self.capacity = capacity_per_street
        self.buffers = defaultdict(lambda: deque(maxlen=capacity_per_street))
        self.lock = threading.Lock()

    def log(self, street, position, q_vector, behavior_pi, valid_mask, summary=""):
        with self.lock:
            self.buffers[street].append({
                'position': position, 'q': q_vector, 'pi': behavior_pi,
                'valid': valid_mask, 'summary': summary,
            })

    def sample_per_street(self, action_names):
        out = []
        with self.lock:
            for street in sorted(self.buffers.keys()):
                buf = self.buffers[street]
                if not buf:
                    continue
                sample = random.choice(list(buf))
                street_name = ['preflop', 'flop', 'turn', 'river'][min(street, 3)]
                out.append(f"\n--- {street_name.upper()} (pos {sample['position']}) {sample['summary']} ---")
                out.append(f"{'Action':<10} | {'pi (behavior)':<14} | {'Q (bb)':<10} | {'valid':<6}")
                out.append("-" * 50)
                for i, name in enumerate(action_names):
                    pi_v = sample['pi'][i]
                    q_v = sample['q'][i]
                    v_v = sample['valid'][i]
                    valid_str = "Y" if v_v else "."
                    out.append(f"{name:<10} | {pi_v:<14.4f} | {q_v:<+10.3f} | {valid_str:<6}")
        return "\n".join(out)


class StreetDiagnostics:
    def __init__(self):
        self.counts = defaultdict(int)
        self.abs_adv_sums = defaultdict(float)
        self.policy_loss_sums = defaultdict(float)
        self.kl_sums = defaultdict(float)
        self.ref_kl_sums = defaultdict(float)
        self.lock = threading.Lock()

    def add(self, street: int, abs_adv: float, policy_loss_contrib: float, kl: float, ref_kl: float = 0.0):
        with self.lock:
            self.counts[street] += 1
            self.abs_adv_sums[street] += abs_adv
            self.policy_loss_sums[street] += policy_loss_contrib
            self.kl_sums[street] += kl
            self.ref_kl_sums[street] += ref_kl

    def get_avg_kl(self, street: int):
        with self.lock:
            n = self.counts[street]
            if n == 0:
                return (None, 0)
            return (self.kl_sums[street] / n, n)

    def report_and_reset(self):
        with self.lock:
            lines = []
            for s in sorted(self.counts.keys()):
                n = self.counts[s]
                if n == 0:
                    continue
                avg_adv = self.abs_adv_sums[s] / n
                avg_loss = self.policy_loss_sums[s] / n
                avg_kl = self.kl_sums[s] / n
                avg_ref_kl = self.ref_kl_sums[s] / n
                lines.append(f"  Street {s}: n={n}, |Adv|={avg_adv:.3f}bb, PolLoss={avg_loss:+.4f}, EMA-KL={avg_kl:.5f}, Ref-KL={avg_ref_kl:.5f}")
            self.counts.clear()
            self.abs_adv_sums.clear()
            self.policy_loss_sums.clear()
            self.kl_sums.clear()
            self.ref_kl_sums.clear()
            return "\n".join(lines)


class GenerationDiagnostics:
    RARITY_EDGES = (0.02, 0.10, 0.30)
    POT_EDGES_BB = (10.0, 30.0, 100.0)

    def __init__(self):
        self.lock = threading.Lock()
        self._reset_locked()

    @staticmethod
    def _bucket(x, edges):
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    def _reset_locked(self):
        self.n = defaultdict(int)
        self.sum_entropy = defaultdict(float)
        self.sum_support = defaultdict(float)
        self.sum_pfold = defaultdict(float)
        self.sum_paggr = defaultdict(float)
        self.gap_n = defaultdict(int)
        self.sum_gap = defaultdict(float)
        self.sum_gap_se = defaultdict(float)
        self.rar_n = defaultdict(int)
        self.rar_se = defaultdict(float)
        self.pot_n = defaultdict(int)
        self.pot_se = defaultdict(float)
        self.scv_n = defaultdict(int)
        self.scv_sum = defaultdict(float)
        self.dom_n = defaultdict(int)
        self.dom_hits = defaultdict(int)
        self.reach_hands = 0
        self.reach_flop = 0
        self.reach_turn = 0
        self.reach_river = 0

    def add_pivot(self, street, entropy, support_size, p_fold, p_aggr, adv_gap=None, adv_gap_se=None, pot_bb=None):
        with self.lock:
            self.n[street] += 1
            self.sum_entropy[street] += float(entropy)
            self.sum_support[street] += float(support_size)
            self.sum_pfold[street] += float(p_fold)
            self.sum_paggr[street] += float(p_aggr)
            if adv_gap is not None and adv_gap_se is not None and np.isfinite(adv_gap_se):
                self.gap_n[street] += 1
                self.sum_gap[street] += float(adv_gap)
                self.sum_gap_se[street] += float(adv_gap_se)
                rb = self._bucket(float(p_aggr), self.RARITY_EDGES)
                self.rar_n[rb] += 1
                self.rar_se[rb] += float(adv_gap_se)
                if pot_bb is not None and np.isfinite(pot_bb):
                    pb = self._bucket(float(pot_bb), self.POT_EDGES_BB)
                    self.pot_n[pb] += 1
                    self.pot_se[pb] += float(adv_gap_se)

    def add_scv(self, street, q_self_minus_ref):
        with self.lock:
            if np.isfinite(q_self_minus_ref):
                self.scv_n[street] += 1
                self.scv_sum[street] += float(q_self_minus_ref)

    def add_dominated(self, street, is_dominated):
        with self.lock:
            self.dom_n[street] += 1
            if is_dominated:
                self.dom_hits[street] += 1

    def add_reach(self, reached_street):
        with self.lock:
            self.reach_hands += 1
            if reached_street >= 1:
                self.reach_flop += 1
            if reached_street >= 2:
                self.reach_turn += 1
            if reached_street >= 3:
                self.reach_river += 1

    def report_and_reset(self):
        with self.lock:
            lines = []
            street_names = ['preflop', 'flop', 'turn', 'river']
            for s in sorted(self.n.keys()):
                n = self.n[s]
                if n == 0:
                    continue
                H = self.sum_entropy[s] / n
                supp = self.sum_support[s] / n
                pf = self.sum_pfold[s] / n
                pa = self.sum_paggr[s] / n
                gn = self.gap_n[s]
                if gn > 0:
                    gap = self.sum_gap[s] / gn
                    gap_se = self.sum_gap_se[s] / gn
                    snr = gap / gap_se if gap_se > 1e-9 else float('inf')
                    gap_str = f"advGapSE={gap_se:.3f}bb advGap={gap:.3f}bb SNR={snr:.2f}"
                else:
                    gap_str = "advGapSE=n/a"
                sname = street_names[min(s, 3)]
                scv_str = ""
                if self.scv_n[s] > 0:
                    scv = self.scv_sum[s] / self.scv_n[s]
                    scv_str = f" | SCV(Qself-Qref)={scv:+.3f}bb(n={self.scv_n[s]})"
                dom_str = ""
                if self.dom_n[s] > 0:
                    dr = self.dom_hits[s] / self.dom_n[s]
                    dom_str = f" | domAct={dr:.1%}(n={self.dom_n[s]})"
                lines.append(f"  {sname:<7} n={n:<4} H={H:.3f} supp={supp:.2f} pFold={pf:.3f} pAggr={pa:.3f} | {gap_str}{scv_str}{dom_str}")

            if sum(self.rar_n.values()) > 0:
                rlabels = ['<2%', '2-10%', '10-30%', '>30%']
                cells = []
                for b in range(len(self.RARITY_EDGES) + 1):
                    if self.rar_n[b] > 0:
                        cells.append(f"{rlabels[b]}:{self.rar_se[b]/self.rar_n[b]:.3f}(n{self.rar_n[b]})")
                lines.append("  advGapSE by p_aggr (rarity): " + "  ".join(cells))
            if sum(self.pot_n.values()) > 0:
                plabels = ['<10bb', '10-30', '30-100', '>100']
                cells = []
                for b in range(len(self.POT_EDGES_BB) + 1):
                    if self.pot_n[b] > 0:
                        cells.append(f"{plabels[b]}:{self.pot_se[b]/self.pot_n[b]:.3f}(n{self.pot_n[b]})")
                lines.append("  advGapSE by pot size:        " + "  ".join(cells))

            if self.reach_hands > 0:
                rh = self.reach_hands
                lines.append(f"  reach: hands={rh} flop={self.reach_flop/rh:.1%} turn={self.reach_turn/rh:.1%} river={self.reach_river/rh:.1%}")
            self._reset_locked()
            return "\n".join(lines)


class Simulator:
    """Manages deep-learning execution loops, game rollouts, and multi-threaded data pipeline generation."""
    def __init__(self):
        self.device = 'cuda'
        torch.set_float32_matmul_precision('high')
        self.thread_local = threading.local()

        config = AutoConfig.from_pretrained('./config.json')
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('models/base_seed.pt', map_location=self.device, weights_only=True), strict=False)
        self.model.share_memory()

        self.ema_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.eval().requires_grad_(False)

        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model.load_state_dict(torch.load('models/base_seed.pt', map_location=self.device, weights_only=True), strict=False)
        self.ref_model.eval().requires_grad_(False)

        _profile_early = os.environ.get('CORE_PROFILE', 'v24').lower()
        self.use_reg_outer_loop = (_profile_early != 'v23')
        if os.environ.get('REG_LOOP_ENABLE', '0') != '1':
            self.use_reg_outer_loop = False

        self.reg_update_period = 1000
        self.eta_reg = 0.02
        if self.use_reg_outer_loop:
            self.reg_model = AutoModelForCausalLM.from_config(config).to(self.device)
            self.reg_model.load_state_dict(self.model.state_dict())
            self.reg_model.eval().requires_grad_(False)
        else:
            self.reg_model = None

        self._reg_iteration = 0
        self._last_reg_update_step = 0

        base_tokenizer = AutoTokenizer.from_pretrained('./model_tokenizer')
        base_tokenizer.padding_side = "left"
        base_tokenizer.pad_token = base_tokenizer.unk_token
        self.unk_token_id = base_tokenizer.unk_token_id

        self.action_names = ['fold', 'check', 'call', 'raise', 'allin']
        self.action_to_idx = {name: i for i, name in enumerate(self.action_names)}
        self.action_tokens = {name: base_tokenizer.encode(f"<{name}>")[0] for name in self.action_names}
        self.action_token_ids_tensor = torch.tensor([self.action_tokens[act] for act in self.action_names], device=self.device)

        self.min_size_token_id = base_tokenizer.encode("<b1%>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        self.sizes = np.array(list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25)), dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        _size_ids = list(range(self.min_size_token_id, self.min_size_token_id + len(self.sizes_floats)))
        self.forbidden_at_grammar_ids = torch.tensor([self.action_tokens[a] for a in self.action_names] + _size_ids, device=self.device)
        self.beta_leak = 1.0

        self.eta_ema = 0.0
        self.eta_ref = 0.0
        self.eta_unif = 0.05
        self.alpha_ent = 0.04
        self.beta_size = 0.2
        self.aux_coef = 0.5
        self.adv_to_logit_scale = 1.0
        self.tau = 5e-4
        self.ppo_clip = 0.30
        self.opp_mix_active = 0.40
        self.opp_mix_ema = 0.40
        self.opp_mix_ref = 0.20
        self.depth_weights = [0.15, 2.0, 5.0, 10.0]
        self.n_sims_by_street = [16, 48, 96, 128]
        self.anchor_clip_bb = 1.0
        self.adv_clip_bb = 3.0
        self.raw_adv_clip_bb = None
        self.beta_mass = 0.5
        self.beta_ref_fwd = 0.08

        self.reach_eps_start = 0.25
        self.reach_eps_floor = 0.10
        self.reach_eps_anneal_steps = 30000
        self.rollout_opp_eps = 0.0
        self.use_crn_rollouts = False
        self.use_reach_reweight = True
        self.reach_reweight_clip = 5.0
        self.enable_subgame_solver = False

        self.scv_probe_enabled = True
        self.scv_probe_prob = 0.10
        self.scv_probe_max_pivots = 4
        self.scv_probe_nsims = 48

        profile = os.environ.get('CORE_PROFILE', 'v24').lower()
        if profile == 'v23':
            self.beta_ref_fwd = 0.0
            self.reach_eps_start = 0.0
            self.reach_eps_floor = 0.0
            self.rollout_opp_eps = 0.0
            self.use_crn_rollouts = False
            self.use_reg_outer_loop = False
            self.enable_subgame_solver = True
        self.active_profile = profile

        self.gen_batch_size = 2
        self.train_batch_size = 4
        self.global_updates = 0

        self.use_avg_model = os.environ.get('DISABLE_AVG_TRACK', '0') != '1'
        if self.use_avg_model:
            self.avg_model = AutoModelForCausalLM.from_config(config)
            self.avg_model.load_state_dict(self.model.state_dict())
            self.avg_model.eval().requires_grad_(False)
        else:
            self.avg_model = None
        self._avg_weight_sum = 0.0

        self.exploiter_enabled = True
        self.exploiter_every = int(os.environ.get('EXP_INTERVAL', '500'))
        self.exploiter_train_updates = int(os.environ.get('EXP_STEPS', '80'))
        self.psro_opponent_prob = float(os.environ.get('PSRO_MIX_PROB', '0.15'))
        self.psro_pool_dir = os.environ.get('PSRO_POOL_PATH', 'pool_storage')
        self.psro_refresh_every = int(os.environ.get('PSRO_REFRESH_INTERVAL', '500'))
        os.makedirs(self.psro_pool_dir, exist_ok=True)
        self.psro_pool = []
        self._in_exploiter = False
        self._exploiter_step = 0
        self._exploiter_idx = 0
        self._protected_state = None
        self._protected_opt_state = None
        self._protected_cfg = None

        self.psro_model = AutoModelForCausalLM.from_config(config)
        self.psro_model.load_state_dict(self.model.state_dict())
        self.psro_model.eval().requires_grad_(False)
        self._psro_on_gpu = False

        self.running_adv_var = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.adv_var_decay = 0.99

        self.subgame_solver = SubgameSolver(
            threshold=0.0005,
            focus_schedule=((3, 100), (2, 200), (1, 300)),
            cooldown_duration=500,
            kl_ema_decay=0.98,
            warmup_steps=500,
            focus_nsims_multiplier=1.5,
            min_samples_before_trigger=100,
        )

        self.role_baseline = RoleBaseline(decay=0.995)
        self.q_pivot_logger = QPivotLogger(capacity_per_street=50)
        self.street_diag = StreetDiagnostics()
        self.gen_diag = GenerationDiagnostics()

        self.gradient_accumulation_steps = 1
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=2e-5,
            warmup_steps=100,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.optimizer.train()

        self.inference_queue = queue.Queue()
        self.ema_inference_queue = queue.Queue()
        self.ref_inference_queue = queue.Queue()
        self.psro_inference_queue = queue.Queue()

    @property
    def tokenizer(self):
        if not hasattr(self.thread_local, 'tokenizer'):
            tok = AutoTokenizer.from_pretrained('./model_tokenizer')
            tok.padding_side = "left"
            tok.pad_token = tok.unk_token
            self.thread_local.tokenizer = tok
        return self.thread_local.tokenizer

    @property
    def encoder(self):
        if not hasattr(self.thread_local, 'encoder'):
            self.thread_local.encoder = CfrSimEncoder()
        return self.thread_local.encoder

    def _run_inference_server(self, srv_queue: queue.Queue, model, name: str):
        model.eval()
        target_batch_size = 4
        while True:
            try:
                req = srv_queue.get()
                reqs = [req]
                total_queries = len(req['queries'])
                while total_queries < target_batch_size:
                    try:
                        new_req = srv_queue.get_nowait()
                        reqs.append(new_req)
                        total_queries += len(new_req['queries'])
                    except queue.Empty:
                        break

                action_reqs = [r for r in reqs if r['type'] == 'action']
                raise_reqs = [r for r in reqs if r['type'] == 'raise']

                all_queries = []
                action_slices = []
                raise_slices = []
                current_pos = 0
                for r in action_reqs:
                    n = len(r['queries'])
                    all_queries.extend(r['queries'])
                    action_slices.append((current_pos, current_pos + n))
                    current_pos += n
                for r in raise_reqs:
                    n = len(r['queries'])
                    all_queries.extend(r['queries'])
                    raise_slices.append((current_pos, current_pos + n))
                    current_pos += n

                if not all_queries:
                    continue

                inputs = self.tokenizer(all_queries, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device, non_blocking=True)
                attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

                with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    action_results = outputs.logits[:, -1, self.action_token_ids_tensor].float()
                    raise_results = outputs.logits[:, -1, :].float()

                for idx, r in enumerate(action_reqs):
                    s, e = action_slices[idx]
                    r['result']['data'] = action_results[s:e]
                    r['event'].set()
                for idx, r in enumerate(raise_reqs):
                    s, e = raise_slices[idx]
                    r['result']['data'] = raise_results[s:e]
                    r['event'].set()
            except Exception as e:
                print(f"Inference Server [{name}] Error: {e}")
                traceback.print_exc()

    def inference_server_loop(self):
        self._run_inference_server(self.inference_queue, self.model, "active")

    def inference_server_loop_ema(self):
        self._run_inference_server(self.ema_inference_queue, self.ema_model, "ema")

    def inference_server_loop_ref(self):
        self._run_inference_server(self.ref_inference_queue, self.ref_model, "ref")

    def inference_server_loop_psro(self):
        self._run_inference_server(self.psro_inference_queue, self.psro_model, "psro")

    def apply_action(self, hand, action, size):
        if action == 'fold':
            hand.fold()
        elif action == 'check':
            hand.check()
        elif action == 'call':
            hand.call()
        elif action == 'raise':
            action_space = hand.get_action_space()
            max_bet = action_space.get('max_bet', 0)
            min_bet = action_space.get('min_bet', 0)
            if size > 0 and size >= (0.5 * max_bet):
                hand.bet_or_raise(max_bet)
            else:
                safe_size = max(min_bet, min(int(round(size)), max_bet))
                hand.bet_or_raise(safe_size)
        elif action == 'allin':
            action_space = hand.get_action_space()
            if 'max_bet' in action_space:
                hand.bet_or_raise(action_space['max_bet'])
            else:
                hand.call()

    @staticmethod
    def detect_street(hand) -> int:
        try:
            board = hand.state.board
        except AttributeError:
            try:
                board = hand.get_u_hand(0)[1]
            except Exception:
                return 0
        n = len(board) if board else 0
        if n == 0:
            return 0
        if n == 3:
            return 1
        if n == 4:
            return 2
        return 3

    @staticmethod
    def _board_of(hand):
        try:
            b = hand.state.board
            return list(b) if b else []
        except Exception:
            try:
                return list(hand.get_u_hand(0)[1]) or []
            except Exception:
                return []

    def _passive_action(self, hand):
        sp = hand.get_action_space()
        if 'check' in sp:
            return ('check', 0)
        if 'call' in sp:
            return ('call', 0)
        return ('fold', 0)

    def _foldout_action(self, hand):
        sp = hand.get_action_space()
        if 'fold' in sp:
            return ('fold', 0)
        if 'check' in sp:
            return ('check', 0)
        return ('call', 0)

    def _drive(self, hand, chooser, max_steps=400, stop_board=None):
        steps = 0
        while not hand.done and steps < max_steps:
            act, sz = chooser(hand)
            self.apply_action(hand, act, sz)
            steps += 1
            if stop_board is not None and len(self._board_of(hand)) >= stop_board:
                break

    def run_engine_selfcheck(self, n_trials=25):
        crn_tested = crn_match = 0
        for _ in range(n_trials):
            try:
                seed = Hand()
                seed.shuffle()
                a = copy.deepcopy(seed)
                b = copy.deepcopy(seed)
                self._drive(a, self._passive_action, stop_board=3)
                self._drive(b, self._passive_action, stop_board=3)
                ba, bb = self._board_of(a), self._board_of(b)
                if len(ba) >= 3 and len(bb) >= 3:
                    crn_tested += 1
                    if ba == bb:
                        crn_match += 1
            except Exception:
                pass
        crn_verdict = f"PASS ({crn_match}/{crn_tested})" if crn_match == crn_tested else "FAIL"

        sd_tested = sd_full = 0
        for _ in range(n_trials):
            try:
                h = Hand()
                h.shuffle()
                self._drive(h, self._passive_action)
                if h.done:
                    sd_tested += 1
                    if self.detect_street(h) == 3:
                        sd_full += 1
            except Exception:
                pass
        fo_tested = fo_empty = 0
        for _ in range(n_trials):
            try:
                h = Hand()
                h.shuffle()
                self._drive(h, self._foldout_action)
                if h.done:
                    fo_tested += 1
                    if len(self._board_of(h)) == 0:
                        fo_empty += 1
            except Exception:
                pass
        acc_verdict = "PASS" if (sd_tested > 0 and sd_full == sd_tested and fo_tested > 0 and fo_empty == fo_tested) else "FAIL"
        print(f"Engine Tests -> CRN: {crn_verdict} | Structure: {acc_verdict}")
        return {'crn': crn_verdict, 'board': acc_verdict}

    def depth_weight_for(self, street: int, focused: bool = False) -> float:
        base = self.depth_weights[min(street, len(self.depth_weights) - 1)]
        if focused:
            fs = self.subgame_solver.current_focus_street()
            if fs is None:
                return base
            return base if street >= fs else 0.0
        return base

    def n_sims_for_street(self, street: int, focused: bool = False) -> int:
        base = self.n_sims_by_street[min(street, len(self.n_sims_by_street) - 1)]
        if focused and street == self.subgame_solver.current_focus_street():
            return int(round(base * self.subgame_solver.focus_nsims_multiplier))
        return base

    @torch.no_grad()
    def select_action_batch(self, hands, temperature=1.0, return_probs=False, opp_source='active', explore_eps=0.0, return_chosen_prob=False):
        if not hands:
            return [], []

        if opp_source == 'ema':
            target_q = self.ema_inference_queue
        elif opp_source == 'ref':
            target_q = self.ref_inference_queue
        elif opp_source == 'psro':
            target_q = self.psro_inference_queue
        else:
            target_q = self.inference_queue

        results = [self.process_hand_cpu(hand) for hand in hands]
        (encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices) = zip(*results)

        action_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]
        raise_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in range(len(hands))]

        evt_action, evt_raise = threading.Event(), threading.Event()
        res_action, res_raise = {}, {}

        target_q.put({'type': 'action', 'queries': action_queries, 'event': evt_action, 'result': res_action})
        target_q.put({'type': 'raise', 'queries': raise_queries, 'event': evt_raise, 'result': res_raise})

        evt_action.wait()
        evt_raise.wait()

        hero_ev_preds = res_action['data']
        raise_logits = res_raise['data']

        pre_sampled_sizes = self._compute_raise_sizes_from_logits(raise_logits, min_bet_tokens, max_bets, pot_sizes)

        batch_size = len(hands)
        logits_mask = torch.full((batch_size, 5), float('-inf'), device=self.device)

        for i in range(batch_size):
            if not can_check[i]:
                logits_mask[i, self.action_to_idx['fold']] = 0.0
            if can_check[i]:
                logits_mask[i, self.action_to_idx['check']] = 0.0
            if can_call[i]:
                if call_is_allin[i]:
                    logits_mask[i, self.action_to_idx['allin']] = 0.0
                else:
                    logits_mask[i, self.action_to_idx['call']] = 0.0
            if can_raise[i]:
                raise_sz = pre_sampled_sizes[i]
                if raise_sz >= max_bets[i]:
                    logits_mask[i, self.action_to_idx['allin']] = 0.0
                else:
                    logits_mask[i, self.action_to_idx['raise']] = 0.0

            if (logits_mask[i] == float('-inf')).all():
                logits_mask[i, self.action_to_idx['fold']] = 0.0

        stable_evs = hero_ev_preds - hero_ev_preds.max(dim=1, keepdim=True).values
        scaled_evs = (stable_evs / temperature) + logits_mask

        probs = torch.softmax(scaled_evs, dim=-1)
        probs = torch.clamp(probs, min=1e-5) * (logits_mask == 0.0).float()
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if explore_eps > 0.0:
            valid_f = (logits_mask == 0.0).float()
            uniform_valid = valid_f / valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            probs = (1.0 - explore_eps) * probs + explore_eps * uniform_valid
            probs = probs / probs.sum(dim=-1, keepdim=True)

        chosen_indices = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()
        final_actions = [self.action_names[idx] for idx in chosen_indices]

        final_sizes = [max_bets[i] if act == 'allin' else (pre_sampled_sizes[i] if act == 'raise' else 0) for i, act in enumerate(final_actions)]

        if return_probs:
            return final_actions, final_sizes, probs.cpu().numpy(), pre_sampled_sizes

        if return_chosen_prob:
            cp = probs[torch.arange(probs.shape[0], device=probs.device), torch.tensor(chosen_indices, device=probs.device)]
            return final_actions, final_sizes, cp.cpu().numpy()

        return final_actions, final_sizes


    @torch.no_grad()
    def evaluate_sampled_actions(self, hands_with_actions, n_sims_per_item=None, crn_groups=None, return_per_scenario=False, force_opp_source=None):
        """Runs downstream simulation rollouts to evaluate state-action values across variance-reduction groups."""
        if n_sims_per_item is None:
            n_sims_per_item = [16] * len(hands_with_actions)

        use_crn = bool(self.use_crn_rollouts and crn_groups is not None)
        offsets = [0]
        for n in n_sims_per_item:
            offsets.append(offsets[-1] + n)
        total_sims = offsets[-1]

        active_sims = []
        finished_payoffs = [None] * total_sims
        big_blinds = [None] * total_sims

        group_seeds = {}
        if use_crn:
            for i, (hand, act, sz, player) in enumerate(hands_with_actions):
                g = crn_groups[i]
                if g not in group_seeds:
                    n_sims_g = n_sims_per_item[i]
                    seeds = []
                    for _ in range(n_sims_g):
                        seed = copy.deepcopy(hand)
                        seed.shuffle()
                        seeds.append(seed)
                    group_seeds[g] = seeds

        for i, (hand, act, sz, player) in enumerate(hands_with_actions):
            bb = max(hand.big_blind, 1)
            n_sims_i = n_sims_per_item[i]
            base = offsets[i]
            seeds_i = group_seeds.get(crn_groups[i]) if use_crn else None
            for j in range(n_sims_i):
                if seeds_i is not None:
                    sim_clone = copy.deepcopy(seeds_i[j])
                else:
                    sim_clone = copy.deepcopy(hand)
                    sim_clone.shuffle()
                orig_idx = base + j
                big_blinds[orig_idx] = bb
                self.apply_action(sim_clone, act, sz)
                if sim_clone.done:
                    p = sim_clone.state.payoffs
                    for x in range(len(p)):
                        if p[x] > 0:
                            p[x] -= min(p[x] * 0.05, 2 * sim_clone.big_blind)
                    finished_payoffs[orig_idx] = p[player]
                else:
                    active_sims.append({'sim': sim_clone, 'orig_idx': orig_idx, 'player': player})

        mix_choices = ['active', 'ema', 'ref']
        mix_probs = [self.opp_mix_active, self.opp_mix_ema, self.opp_mix_ref]
        if getattr(self, 'psro_opponent_prob', 0.0) > 0.0 and self.psro_pool and not self._in_exploiter:
            pp = float(self.psro_opponent_prob)
            mix_choices = ['active', 'ema', 'ref', 'psro']
            mix_probs = [self.opp_mix_active * (1 - pp), self.opp_mix_ema * (1 - pp), self.opp_mix_ref * (1 - pp), pp]
        
        n_src = len(mix_choices)
        s = sum(mix_probs)
        mix_probs = [p / s for p in mix_probs]
        forced_idx = mix_choices.index(force_opp_source) if force_opp_source is not None else None

        while active_sims:
            if forced_idx is not None:
                sources_idx = np.full(len(active_sims), forced_idx, dtype=int)
            else:
                sources_idx = np.random.choice(n_src, size=len(active_sims), p=mix_probs)
            groups = {i: [] for i in range(n_src)}
            for k, src_idx in enumerate(sources_idx):
                groups[src_idx].append((k, active_sims[k]))

            actions_by_k = {}
            sizes_by_k = {}
            for src_idx, group_items in groups.items():
                if not group_items:
                    continue
                src_name = mix_choices[src_idx]
                group_sims = [item[1]['sim'] for item in group_items]
                acts, szs = self.select_action_batch(group_sims, temperature=1.0, opp_source=src_name, explore_eps=self.rollout_opp_eps)
                for (k, _), act, sz in zip(group_items, acts, szs):
                    actions_by_k[k] = act
                    sizes_by_k[k] = sz

            next_active = []
            for k, sim_dict in enumerate(active_sims):
                act = actions_by_k[k]
                sz = sizes_by_k[k]
                sim = sim_dict['sim']
                self.apply_action(sim, act, sz)
                if sim.done:
                    p = sim.state.payoffs
                    for x in range(len(p)):
                        if p[x] > 0:
                            p[x] -= min(p[x] * 0.05, 2 * sim.big_blind)
                    finished_payoffs[sim_dict['orig_idx']] = p[sim_dict['player']]
                else:
                    next_active.append(sim_dict)
            active_sims = next_active

        rewards_bb = []
        per_scenario_bb = []
        for i in range(len(hands_with_actions)):
            payoffs = finished_payoffs[offsets[i]:offsets[i + 1]]
            bbs = big_blinds[offsets[i]:offsets[i + 1]]
            payoffs_bb = [p / b for p, b in zip(payoffs, bbs)]
            rewards_bb.append(float(np.mean(payoffs_bb)))
            if return_per_scenario:
                per_scenario_bb.append(np.asarray(payoffs_bb, dtype=np.float64))
        return (rewards_bb, per_scenario_bb) if return_per_scenario else rewards_bb

    def _reach_eps_now(self):
        s, f, n = float(self.reach_eps_start), float(self.reach_eps_floor), int(self.reach_eps_anneal_steps)
        if n <= 0:
            return f
        frac = min(1.0, max(0, self.global_updates) / float(n))
        return max(f, s + (f - s) * frac)

    def generate_rnad_trajectories(self, train_count):
        focused = self.subgame_solver.in_focus()
        hands = [Hand() for _ in range(self.gen_batch_size)]
        hero_indices = [randint(0, 5) for _ in range(self.gen_batch_size)]
        hero_snapshots = [[] for _ in range(self.gen_batch_size)]
        active_indices = list(range(self.gen_batch_size))
        reach_logprob = [0.0] * self.gen_batch_size

        with global_profiler.profile("Worker: Generate Prefixes"):
            while active_indices:
                current_hands = [hands[i] for i in active_indices]
                is_hero_turn = [False] * len(active_indices)
                for i, idx in enumerate(active_indices):
                    hand = hands[idx]
                    if not hand.done and hand.state.turn_index == hero_indices[idx]:
                        is_hero_turn[i] = True
                        street = self.detect_street(hand)
                        if focused and self.depth_weight_for(street, focused=True) <= 0.0:
                            continue
                        hero_snapshots[idx].append((copy.deepcopy(hand), street, reach_logprob[idx]))

                actions, sizes, chosen_probs = self.select_action_batch(
                    current_hands, temperature=1.0, opp_source='active',
                    explore_eps=self._reach_eps_now(), return_chosen_prob=True
                )
                next_active = []
                for i, (action, size) in enumerate(zip(actions, sizes)):
                    hand_idx = active_indices[i]
                    self.apply_action(hands[hand_idx], action, size)
                    if is_hero_turn[i]:
                        reach_logprob[hand_idx] += math.log(max(float(chosen_probs[i]), 1e-6))
                    if not hands[hand_idx].done:
                        next_active.append(hand_idx)
                active_indices = next_active

        try:
            for idx in range(self.gen_batch_size):
                self.gen_diag.add_reach(self.detect_street(hands[idx]))
        except Exception:
            pass

        pivot_hands, pivot_heroes, pivot_streets, pivot_reach_lp = [], [], [], []
        for idx in range(self.gen_batch_size):
            for snap, street, rlp in hero_snapshots[idx]:
                pivot_hands.append(snap)
                pivot_heroes.append(hero_indices[idx])
                pivot_streets.append(street)
                pivot_reach_lp.append(rlp)

        if not pivot_hands:
            return []

        max_pivots = 12
        if focused:
            keep = [i for i, s in enumerate(pivot_streets) if self.depth_weight_for(s, focused=True) > 0.0]
            pivot_hands = [pivot_hands[i] for i in keep]
            pivot_heroes = [pivot_heroes[i] for i in keep]
            pivot_streets = [pivot_streets[i] for i in keep]
            pivot_reach_lp = [pivot_reach_lp[i] for i in keep]

        if not pivot_hands:
            return []

        if len(pivot_hands) > max_pivots:
            weights = np.array([self.depth_weight_for(s, focused=focused) for s in pivot_streets], dtype=np.float64)
            wsum = weights.sum()
            if wsum < 1e-8:
                return []
            weights = weights / wsum
            chosen = np.random.choice(len(pivot_hands), size=max_pivots, replace=False, p=weights)
            pivot_hands = [pivot_hands[i] for i in chosen]
            pivot_heroes = [pivot_heroes[i] for i in chosen]
            pivot_streets = [pivot_streets[i] for i in chosen]
            pivot_reach_lp = [pivot_reach_lp[i] for i in chosen]

        with global_profiler.profile("Worker: Compute Pivot Priors"):
            _, _, batch_probs, pre_sampled_sizes = self.select_action_batch(pivot_hands, return_probs=True, opp_source='active')

        hands_for_eval, eval_map, valid_actions_per_hand, n_sims_per_eval = [], [], [], []
        for i, hand in enumerate(pivot_hands):
            sz = pre_sampled_sizes[i]
            hero_idx = pivot_heroes[i]
            street = pivot_streets[i]
            n_sims_this_pivot = self.n_sims_for_street(street, focused=focused)
            action_space = hand.get_action_space()

            can_check = 'check' in action_space
            can_raise = 'min_bet' in action_space
            max_bet = action_space.get('max_bet', 0)
            if can_raise and action_space['min_bet'] >= max_bet:
                can_raise = False
            can_call = 'call' in action_space
            call_is_allin = can_call and not can_raise

            valid_acts = ['check' if can_check else 'fold']
            if can_call:
                valid_acts.append('allin' if call_is_allin else 'call')
            if can_raise:
                valid_acts.append('allin' if sz >= max_bet else 'raise')

            valid_actions_per_hand.append(valid_acts)
            for act in valid_acts:
                hands_for_eval.append((copy.deepcopy(hand), act, sz if act == 'raise' else 0, hero_idx))
                eval_map.append((i, act, sz if act == 'raise' else 0))
                n_sims_per_eval.append(n_sims_this_pivot)

        with global_profiler.profile("Worker: Mixed-opponent CFR Rollouts"):
            crn_groups = [m[0] for m in eval_map]
            rewards, per_scenario = self.evaluate_sampled_actions(hands_for_eval, n_sims_per_item=n_sims_per_eval, crn_groups=crn_groups, return_per_scenario=True)

        q_values_per_hand = [{} for _ in range(len(pivot_hands))]
        sizes_per_hand = [{} for _ in range(len(pivot_hands))]
        scenario_by_pivot = [{} for _ in range(len(pivot_hands))]
        for e_idx, reward in enumerate(rewards):
            pivot_idx, act, sz = eval_map[e_idx]
            q_values_per_hand[pivot_idx][act] = reward
            sizes_per_hand[pivot_idx][act] = sz
            scenario_by_pivot[pivot_idx][act] = per_scenario[e_idx]

        if self.scv_probe_enabled and not focused and random.random() < self.scv_probe_prob:
            try:
                cand = [(i, 'raise' if 'raise' in va else 'call') for i, va in enumerate(valid_actions_per_hand) if 'raise' in va or 'call' in va]
                random.shuffle(cand)
                cand = cand[:self.scv_probe_max_pivots]
                if cand:
                    probe_hands = [(copy.deepcopy(pivot_hands[i]), cont, pre_sampled_sizes[i] if cont == 'raise' else 0, pivot_heroes[i]) for i, cont in cand]
                    ns = [self.scv_probe_nsims] * len(probe_hands)
                    q_self = self.evaluate_sampled_actions(probe_hands, n_sims_per_item=ns, force_opp_source='ema')
                    q_ref = self.evaluate_sampled_actions(probe_hands, n_sims_per_item=ns, force_opp_source='ref')
                    for (i, _), qs, qr in zip(cand, q_self, q_ref):
                        self.gen_diag.add_scv(pivot_streets[i], float(qs) - float(qr))
            except Exception:
                pass

        experiences = []
        pending = []

        for i, hand in enumerate(pivot_hands):
            hero_idx = pivot_heroes[i]
            street = pivot_streets[i]
            q_dict = q_values_per_hand[i]
            valid_acts = valid_actions_per_hand[i]
            behavior_probs = batch_probs[i]

            q_arr = np.zeros(5, dtype=np.float32)
            valid_mask = np.zeros(5, dtype=bool)
            for act in valid_acts:
                a_idx = self.action_to_idx[act]
                q_arr[a_idx] = q_dict[act]
                valid_mask[a_idx] = True

            v_now_bb = float(np.sum(behavior_probs * np.where(valid_mask, q_arr, 0.0)))
            self.role_baseline.update(street, hero_idx, v_now_bb)

            try:
                bp = np.asarray(behavior_probs, dtype=np.float64)
                vm = valid_mask.astype(bool)
                p_valid = bp[vm] / max(bp[vm].sum(), 1e-12)
                entropy = float(-(p_valid * np.log(p_valid + 1e-12)).sum())
                support_size = int((p_valid > 0.01).sum())
                p_fold = float(bp[self.action_to_idx['fold']]) if vm[self.action_to_idx['fold']] else 0.0
                p_aggr = float((bp[self.action_to_idx['raise']] if vm[self.action_to_idx['raise']] else 0.0) + (bp[self.action_to_idx['allin']] if vm[self.action_to_idx['allin']] else 0.0))

                adv_gap = adv_gap_se = None
                sc = scenario_by_pivot[i]
                if len(sc) >= 2:
                    means = {a: float(np.mean(arr)) for a, arr in sc.items()}
                    ranked = sorted(means, key=means.get, reverse=True)
                    m = min(len(sc[ranked[0]]), len(sc[ranked[1]]))
                    if m >= 2:
                        d = sc[ranked[0]][:m] - sc[ranked[1]][:m]
                        adv_gap, adv_gap_se = float(abs(d.mean())), float(d.std(ddof=1) / math.sqrt(m))
                pot_bb = float(hand.pot_size()) / max(float(hand.big_blind), 1.0)
                self.gen_diag.add_pivot(street, entropy, support_size, p_fold, p_aggr, adv_gap=adv_gap, adv_gap_se=adv_gap_se, pot_bb=pot_bb)
            except Exception:
                pass

            try:
                u = hand.get_u_hand(hero_idx)
                summary = f"hole={u[0][hero_idx] if len(u[0]) > hero_idx else '??'} board={''.join(u[1])[:10] if len(u) > 1 else ''}"
            except Exception:
                summary = ""
            self.q_pivot_logger.log(street=street, position=hero_idx, q_vector=q_arr.tolist(), behavior_pi=list(behavior_probs), valid_mask=valid_mask.tolist(), summary=summary)

            valid_probs = behavior_probs * valid_mask.astype(np.float32)
            a_choice_idx = int(np.random.choice(5, p=valid_probs / valid_probs.sum())) if valid_probs.sum() > 1e-6 else random.choice([self.action_to_idx[a] for a in valid_acts])

            represent_act = self.action_names[a_choice_idx]
            represent_sz = sizes_per_hand[i].get(represent_act, 0)

            pre_action_str = self.encoder.encode(json.dumps(hand.get_u_hand(hero_idx)), explicit_hero_idx=hero_idx, include_result=False, strict_rl_alignment=True)
            pivot_marker = f"<herop{hero_idx}>"
            target_occurrence = pre_action_str.count(pivot_marker) + 1

            train_hand = copy.deepcopy(hand)
            self.apply_action(train_hand, represent_act, represent_sz)
            depth_w = self.depth_weight_for(street, focused=focused)

            dominated = bool(int(np.argmax(np.where(valid_mask.astype(bool), behavior_probs, -1.0))) != int(np.argmax(np.where(valid_mask.astype(bool), q_arr, -np.inf)))) if valid_mask.any() else None
            if dominated is not None:
                try:
                    self.gen_diag.add_dominated(street, dominated)
                except Exception:
                    pass

            pending.append({
                'train_hand': train_hand, 'hero_idx': hero_idx, 'street': street, 'q_arr': q_arr,
                'valid_mask': valid_mask, 'behavior_probs': behavior_probs, 'represent_act': represent_act,
                'pivot_marker': pivot_marker, 'target_occurrence': target_occurrence, 'depth_w': float(depth_w),
                'reach_lp': float(pivot_reach_lp[i]),
            })

        with global_profiler.profile("Worker: Batched Representative Rollout"):
            active = [p['train_hand'] for p in pending if not p['train_hand'].done]
            while active:
                acts, szs = self.select_action_batch(active, temperature=1.0, opp_source='ema', explore_eps=self.rollout_opp_eps)
                for h, a, s in zip(active, acts, szs):
                    self.apply_action(h, a, s)
                active = [h for h in active if not h.done]

        for p in pending:
            final_text = self.encoder.encode(json.dumps(p['train_hand'].get_u_hand(p['hero_idx'])), explicit_hero_idx=p['hero_idx'], include_result=True, strict_rl_alignment=True)
            final_ids = self.tokenizer(final_text).input_ids
            pivot_token_id = self.tokenizer.encode(p['pivot_marker'], add_special_tokens=False)[-1]

            occurrence_count = 0
            action_target_idx = -1
            for t_idx in range(len(final_ids)):
                if final_ids[t_idx] == pivot_token_id:
                    occurrence_count += 1
                    if occurrence_count == p['target_occurrence']:
                        action_target_idx = t_idx + 1
                        break

            if action_target_idx == -1 or action_target_idx >= len(final_ids):
                continue

            experiences.append({
                'full_text': final_text, 'train_count': train_count, 'action_idx': action_target_idx,
                'q_values_bb': p['q_arr'].tolist(), 'valid_mask': p['valid_mask'].tolist(), 'behavior_pi': p['behavior_probs'].tolist(),
                'depth_weight': p['depth_w'], 'reach_lp': p['reach_lp'], 'street': int(p['street']),
                'represent_action': p['represent_act'], 'generated_in_focus': bool(focused),
            })
        return experiences

    def worker_loop(self, worker_id, q, train_count_obj):
        self.model.eval()
        gc.disable()
        while True:
            try:
                if train_count_obj.get() == train_count_obj.get():
                    with torch.no_grad():
                        experiences = self.generate_rnad_trajectories(train_count_obj.get())
                    for exp in experiences:
                        q.put(exp)
                gc.collect()
            except Exception as e:
                print(f"Generator {worker_id} Exception: {e}")
                traceback.print_exc()

    def _anchor_cfg_snapshot(self):
        return dict(
            eta_ema=self.eta_ema, eta_ref=self.eta_ref, eta_unif=self.eta_unif,
            alpha_ent=self.alpha_ent, beta_ref_fwd=self.beta_ref_fwd,
            opp_mix_active=self.opp_mix_active, opp_mix_ema=self.opp_mix_ema,
            opp_mix_ref=self.opp_mix_ref, psro_opponent_prob=self.psro_opponent_prob,
            use_reg_outer_loop=self.use_reg_outer_loop,
        )

    def _anchor_cfg_restore(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

    def _enter_exploiter_phase(self, train_count_obj):
        self._exploiter_idx += 1
        print(f"\n*** [EXPLOITER PHASE #{self._exploiter_idx}] entering at update {self.global_updates} ***")
        self._protected_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            self._protected_opt_state = copy.deepcopy(self.optimizer.state_dict())
        except Exception:
            self._protected_opt_state = None
        self._protected_cfg = self._anchor_cfg_snapshot()
        
        with torch.no_grad():
            self.ref_model.load_state_dict(self._protected_state)
            
        self.eta_ema = 0.0; self.eta_ref = 0.0; self.eta_unif = 0.0
        self.beta_ref_fwd = 0.0; self.use_reg_outer_loop = False
        self.opp_mix_active = 0.0; self.opp_mix_ema = 0.0; self.opp_mix_ref = 1.0
        self.psro_opponent_prob = 0.0
        self._in_exploiter = True
        self._exploiter_step = 0
        train_count_obj.increment()

    def _exit_exploiter_phase(self, train_count_obj):
        path = os.path.join(self.psro_pool_dir, f"exploiter-{self._exploiter_idx:03d}-at{self.global_updates}.pt")
        torch.save(self.model.state_dict(), path)
        self.psro_pool.append(path)
        
        with torch.no_grad():
            self.model.load_state_dict(self._protected_state)
        if self._protected_opt_state is not None:
            try:
                self.optimizer.load_state_dict(self._protected_opt_state)
            except Exception:
                pass
        self._anchor_cfg_restore(self._protected_cfg)
        with torch.no_grad():
            self.ref_model.load_state_dict(torch.load('models/base_seed.pt', map_location=self.device, weights_only=True))
        self._in_exploiter = False
        self._protected_state = None
        self._protected_opt_state = None
        self.refresh_psro_opponent(prefer_latest=True)
        train_count_obj.increment()

    def refresh_psro_opponent(self, prefer_latest=False):
        if not self.psro_pool:
            return
        if not self._psro_on_gpu:
            self.psro_model.to(self.device)
            self._psro_on_gpu = True
        path = self.psro_pool[-1] if prefer_latest else random.choice(self.psro_pool)
        try:
            with torch.no_grad():
                self.psro_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.psro_model.eval()
        except Exception as e:
            print(f"PSRO load failure {path}: {e}")

    def run_trainer(self, q, train_count_obj):
        """Primary optimizer stepping sequence executing game-theory policy gradient updates over raw data frames."""
        try:
            self.run_engine_selfcheck()
        except Exception as e:
            print(f"Self-check missing/halted: {e}")

        log_file = open('training_logs.csv', mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([
            'Update', 'Total_Loss', 'Pol_Loss', 'Ent_Loss', 'Size_Loss', 'Gram_Loss',
            'EMA_KL', 'Ref_KL', 'Unif_KL', 'Preflop_KL', 'Mean_AbsAdv', 'Mean_AbsQ',
            'MinP', 'Clip_Frac', 'SG_State', 'SG_PhaseRem', 'SG_Activations',
        ])

        batch_memory = []
        losses = {
            'total': [], 'policy': [], 'entropy': [], 'size': [], 'grammar': [],
            'ema_kl': [], 'ref_kl': [], 'unif_kl': [], 'preflop_kl': [],
            'abs_adv': [], 'abs_q': [], 'min_prob': [], 'clip_frac': [],
            'ref_fwd': [], 'reg_kl': [],
        }
        step_count = 0
        acc_losses = {k: 0.0 for k in losses.keys()}

        train_tokenizer = AutoTokenizer.from_pretrained('./model_tokenizer')
        train_tokenizer.padding_side = "right"
        train_tokenizer.pad_token = train_tokenizer.unk_token
        self.optimizer.zero_grad()

        while self.global_updates < 1_000_000:
            with global_profiler.profile("Trainer: Waiting for Queue"):
                while len(batch_memory) < self.train_batch_size:
                    try:
                        exp = q.get(timeout=1)
                        if exp['train_count'] == train_count_obj.get():
                            batch_memory.append(exp)
                    except queue.Empty:
                        continue

            batch_samples = batch_memory[:self.train_batch_size]
            batch_memory = batch_memory[self.train_batch_size:]

            inputs = train_tokenizer([s['full_text'] for s in batch_samples], padding=True, max_length=256, truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

            q_bb = torch.tensor([s['q_values_bb'] for s in batch_samples], device=self.device, dtype=torch.float32)
            valid_masks_bool = torch.tensor([s['valid_mask'] for s in batch_samples], device=self.device, dtype=torch.bool)
            behavior_pi = torch.tensor([s['behavior_pi'] for s in batch_samples], device=self.device, dtype=torch.float32)
            depth_weights = torch.tensor([s['depth_weight'] for s in batch_samples], device=self.device, dtype=torch.float32)
            action_indices = torch.tensor([s['action_idx'] for s in batch_samples], device=self.device)
            streets = torch.tensor([s['street'] for s in batch_samples], device=self.device)

            B = len(batch_samples)

            if self.use_reach_reweight:
                reach_lp = torch.tensor([s.get('reach_lp', 0.0) for s in batch_samples], device=self.device, dtype=torch.float32)
                log_raw = -reach_lp
                logC = math.log(float(self.reach_reweight_clip))
                rw = torch.ones(B, device=self.device, dtype=torch.float32)
                for s_val in torch.unique(streets):
                    m = (streets == s_val)
                    if m.sum() <= 1:
                        continue
                    u = torch.exp((log_raw[m] - log_raw[m].mean()).clamp(min=-logC, max=logC))
                    rw[m] = u / u.mean().clamp(min=1e-6)
                depth_weights = depth_weights * rw

            self.model.train()
            with global_profiler.profile("Trainer: Forward & Backward R-NaD v6.4"):
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    with torch.no_grad():
                        ema_outputs = self.ema_model(input_ids, attention_mask=attention_mask)
                        ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                        reg_outputs = self.reg_model(input_ids, attention_mask=attention_mask) if self.use_reg_outer_loop else None

                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_ema_logits = ema_outputs.logits[..., :-1, :].contiguous()
                    shift_ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    pivot_pos = (action_indices - 1)
                    b_arange = torch.arange(B, device=self.device)
                    
                    theta_pivot_act = shift_logits[b_arange, pivot_pos][:, self.action_token_ids_tensor]
                    ema_pivot_act = shift_ema_logits[b_arange, pivot_pos][:, self.action_token_ids_tensor]
                    ref_pivot_act = shift_ref_logits[b_arange, pivot_pos][:, self.action_token_ids_tensor]

                    valid_logit_mask = torch.where(valid_masks_bool, torch.zeros_like(theta_pivot_act), torch.full_like(theta_pivot_act, float('-inf')))
                    theta_log = torch.nan_to_num(F.log_softmax(theta_pivot_act + valid_logit_mask, dim=-1), nan=-15.0, neginf=-15.0)
                    ema_log = torch.nan_to_num(F.log_softmax(ema_pivot_act + valid_logit_mask, dim=-1), nan=-15.0, neginf=-15.0)
                    ref_log = torch.nan_to_num(F.log_softmax(ref_pivot_act + valid_logit_mask, dim=-1), nan=-15.0, neginf=-15.0)

                    if self.use_reg_outer_loop:
                        reg_pivot_act = reg_outputs.logits[..., :-1, :].contiguous()[b_arange, pivot_pos][:, self.action_token_ids_tensor] + valid_logit_mask
                        reg_log = torch.nan_to_num(F.log_softmax(reg_pivot_act, dim=-1), nan=-15.0, neginf=-15.0)
                    else:
                        reg_log = ema_log

                    valid_mask_f = valid_masks_bool.float()
                    behavior_pi_norm = (behavior_pi * valid_mask_f) / (behavior_pi * valid_mask_f).sum(dim=1, keepdim=True).clamp(min=1e-8)

                    adv_raw = q_bb - (behavior_pi_norm * q_bb).sum(dim=1, keepdim=True)

                    with torch.no_grad():
                        per_sample_sq = (adv_raw ** 2 * valid_mask_f).sum(dim=1)
                        per_sample_cnt = valid_mask_f.sum(dim=1).clamp(min=1.0)
                        std_per_sample = torch.full_like(per_sample_sq, max(math.sqrt(self.running_adv_var[-1]), 1.0))
                        for s in range(len(self.running_adv_var)):
                            sel = (streets == s)
                            if sel.any():
                                batch_var_s = (per_sample_sq[sel].sum() / per_sample_cnt[sel].sum().clamp(min=1.0)).item()
                                self.running_adv_var[s] = (self.adv_var_decay * self.running_adv_var[s] + (1.0 - self.adv_var_decay) * batch_var_s)
                            std_per_sample = torch.where(sel, torch.full_like(std_per_sample, max(math.sqrt(self.running_adv_var[s]), 1.0)), std_per_sample)
                        global_std = std_per_sample.unsqueeze(1)

                    adv_normalized = adv_raw / global_std
                    theta_log_d, ema_log_d, ref_log_d, reg_log_d = theta_log.detach(), ema_log.detach(), ref_log.detach(), reg_log.detach()

                    num_valid = valid_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
                    log_uniform = -torch.log(num_valid).expand_as(theta_log_d) * valid_mask_f

                    ema_anchor = (self.eta_ema * (theta_log_d - ema_log_d)).clamp(min=-self.anchor_clip_bb, max=self.anchor_clip_bb)
                    ref_anchor = (self.eta_ref * (theta_log_d - ref_log_d)).clamp(min=-self.anchor_clip_bb, max=self.anchor_clip_bb)
                    unif_anchor = (self.eta_unif * (theta_log_d - log_uniform)).clamp(min=-self.anchor_clip_bb, max=self.anchor_clip_bb)
                    primary_anchor = (self.eta_reg * (theta_log_d - reg_log_d)).clamp(min=-self.anchor_clip_bb, max=self.anchor_clip_bb) if self.use_reg_outer_loop else ema_anchor

                    adv_reg = torch.clamp(adv_normalized - primary_anchor - ref_anchor - unif_anchor, min=-self.adv_clip_bb, max=self.adv_clip_bb) * valid_mask_f
                    adv_reg_d = adv_reg.detach()

                    log_ratio = theta_log - ema_log_d
                    clip_lo, clip_hi = math.log(1.0 - self.ppo_clip), math.log(1.0 + self.ppo_clip)
                    clip_frac = ((log_ratio < clip_lo) | (log_ratio > clip_hi)).float().mean()
                    
                    dw_sum = depth_weights.sum().clamp(min=1e-6)
                    policy_loss = ((-( (valid_mask_f / num_valid) * adv_reg_d * (ema_log_d + torch.clamp(log_ratio, min=clip_lo, max=clip_hi)) ).sum(dim=1) * depth_weights).sum() / dw_sum)

                    log_theta_act_mass = torch.logsumexp(F.log_softmax(shift_logits[b_arange, pivot_pos].float(), dim=-1)[:, self.action_token_ids_tensor] + valid_logit_mask, dim=-1)
                    with torch.no_grad():
                        log_ref_act_mass = torch.logsumexp(F.log_softmax(shift_ref_logits[b_arange, pivot_pos].float(), dim=-1)[:, self.action_token_ids_tensor] + valid_logit_mask, dim=-1)
                    mass_anchor_loss = self.beta_mass * ((torch.nan_to_num(log_theta_act_mass - log_ref_act_mass, nan=0.0).pow(2) * depth_weights).sum() / dw_sum)

                    entropy_loss = -self.alpha_ent * ((-(torch.exp(theta_log) * theta_log * valid_mask_f).sum(dim=1) * depth_weights).sum() / dw_sum)

                    if self.beta_ref_fwd > 0.0:
                        ref_fwd_per_sample = (torch.exp(ref_log).detach() * torch.nan_to_num(ref_log.detach() - theta_log, nan=0.0)).sum(dim=1).clamp(min=0.0)
                        ref_fwd_loss = self.beta_ref_fwd * ((ref_fwd_per_sample * depth_weights).sum() / dw_sum)
                    else:
                        ref_fwd_loss = torch.zeros((), device=shift_logits.device)

                    with torch.no_grad():
                        pi_pivot_d = torch.exp(theta_log_d).float()
                        per_sample_kl = (pi_pivot_d * (theta_log_d - ema_log_d)).sum(dim=-1)
                        per_sample_ref_kl = (pi_pivot_d * (theta_log_d - ref_log_d)).sum(dim=-1)
                        
                        per_sample_abs_adv = (adv_raw.abs() * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)
                        for b in range(B):
                            self.street_diag.add(int(streets[b].item()), per_sample_abs_adv[b].item(), policy_loss.item(), per_sample_kl[b].item(), per_sample_ref_kl[b].item())

                    valid_tokens = attention_mask[..., 1:].contiguous().float()
                    size_mask, grammar_mask = torch.zeros_like(shift_labels, dtype=torch.float32), torch.ones_like(shift_labels, dtype=torch.float32)

                    for b in range(B):
                        action_idx_b = batch_samples[b]['action_idx'] - 1
                        seq_len = valid_tokens[b].sum().int().item()
                        if action_idx_b < seq_len:
                            grammar_mask[b, action_idx_b] = 0.0
                            if batch_samples[b]['represent_action'] in ['raise', 'call', 'allin'] and (action_idx_b + 1) < seq_len:
                                size_mask[b, action_idx_b + 1] = 1.0
                                grammar_mask[b, action_idx_b + 1] = 0.0

                    size_mask, grammar_mask = size_mask * valid_tokens, grammar_mask * valid_tokens
                    sym_kl = (torch.exp(F.log_softmax(shift_ref_logits, dim=-1)).detach() * torch.nan_to_num(F.log_softmax(shift_ref_logits, dim=-1) - F.log_softmax(shift_logits, dim=-1), nan=0.0)).sum(dim=-1) + self.beta_leak * (torch.exp(F.log_softmax(shift_logits, dim=-1)) * torch.nan_to_num(F.log_softmax(shift_logits, dim=-1) - F.log_softmax(shift_ref_logits, dim=-1), nan=0.0)).sum(dim=-1)

                    size_total = (self.beta_size * sym_kl * size_mask).sum() / torch.clamp(size_mask.sum(), min=1.0)
                    grammar_total = (self.aux_coef * sym_kl * grammar_mask).sum() / torch.clamp(grammar_mask.sum(), min=1.0)
                    leak_total = torch.zeros((), device=shift_logits.device)

                    with torch.no_grad():
                        ema_kl_div = per_sample_kl.mean()
                        ref_kl_div = per_sample_ref_kl.mean()
                        reg_kl_div = (pi_pivot_d * (theta_log_d - reg_log_d)).sum(dim=-1).mean()
                        unif_kl_div = (pi_pivot_d * valid_mask_f * (theta_log_d - log_uniform)).sum(dim=-1).mean()
                        abs_adv = (adv_raw.abs() * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
                        abs_q = (q_bb.abs() * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
                        min_action_prob = pi_pivot_d.masked_fill(~valid_masks_bool, 1.0).min(dim=1).values.mean()

                        preflop_mask = (streets == 0).float()
                        preflop_kl_batch = ((per_sample_kl * preflop_mask).sum() / preflop_mask.sum().item()).item() if preflop_mask.sum().item() > 0 else None
                        if preflop_kl_batch is not None:
                            self.subgame_solver.record_preflop_kl(preflop_kl_batch)

                    loss = policy_loss + entropy_loss + size_total + grammar_total + mass_anchor_loss + leak_total + ref_fwd_loss

                if torch.isnan(loss):
                    self.optimizer.zero_grad()
                    continue

                (loss / self.gradient_accumulation_steps).backward()

            acc_losses['total'] += loss.item()
            acc_losses['policy'] += policy_loss.item()
            acc_losses['entropy'] += entropy_loss.item()
            acc_losses['size'] += size_total.item()
            acc_losses['grammar'] += grammar_total.item()
            acc_losses['ref_fwd'] += ref_fwd_loss.item()
            acc_losses['reg_kl'] += reg_kl_div.item()
            acc_losses['ema_kl'] += ema_kl_div.item()
            acc_losses['ref_kl'] += ref_kl_div.item()
            acc_losses['unif_kl'] += unif_kl_div.item()
            if preflop_kl_batch is not None:
                acc_losses['preflop_kl'] += preflop_kl_batch
            acc_losses['abs_adv'] += abs_adv.item()
            acc_losses['abs_q'] += abs_q.item()
            acc_losses['min_prob'] += min_action_prob.item()
            acc_losses['clip_frac'] += clip_frac.item()

            step_count += 1
            if step_count % self.gradient_accumulation_steps == 0:
                with global_profiler.profile("Trainer: Optimizer Step"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                            ema_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

                        if self.use_avg_model and not self._in_exploiter:
                            t_w = float(max(self.global_updates, 1))
                            self._avg_weight_sum += t_w
                            alpha = t_w / self._avg_weight_sum
                            for param, avg_param in zip(self.model.parameters(), self.avg_model.parameters()):
                                avg_param.data.mul_(1.0 - alpha).add_(param.data.detach().to('cpu', copy=True), alpha=alpha)

                for k in losses.keys():
                    losses[k].append(acc_losses[k] / self.gradient_accumulation_steps)
                    acc_losses[k] = 0.0

                if self._in_exploiter:
                    self._exploiter_step += 1
                    if self._exploiter_step >= self.exploiter_train_updates:
                        self._exit_exploiter_phase(train_count_obj)
                    step_count += 1
                    continue

                self.global_updates += 1

                if self.use_reg_outer_loop and (self.global_updates - self._last_reg_update_step) >= self.reg_update_period:
                    with torch.no_grad():
                        self.reg_model.load_state_dict(self.model.state_dict())
                    self._reg_iteration += 1
                    self._last_reg_update_step = self.global_updates
                    print(f"\n*** [Outer Loop Snapshot] updated step {self.global_updates} (iteration #{self._reg_iteration}) ***\n")

                transition = self.subgame_solver.tick(self.global_updates)
                if transition is not None and transition[0] == 'released':
                    print(f"*** Subgame solver released at step {transition[1]} ***")

                if self.enable_subgame_solver and self.global_updates % 25 == 0:
                    if self.subgame_solver.maybe_trigger(self.global_updates):
                        print(f"*** Subgame solver active at step {self.global_updates} ***")

                if self.global_updates % 50 == 0:
                    global_profiler.report_and_reset(self.global_updates)
                    print(f"Update {self.global_updates} | Loss: {float(np.mean(losses['total'])):+.4f} | Adv: {float(np.mean(losses['abs_adv'])):.3f} | Clip: {float(np.mean(losses['clip_frac'])):.1%} | ReachEps: {self._reach_eps_now():.3f}")
                    
                    street_report = self.street_diag.report_and_reset()
                    if street_report:
                        print(street_report)
                    gen_report = self.gen_diag.report_and_reset()
                    if gen_report:
                        print(gen_report)

                    for k in losses.keys():
                        losses[k] = []

                if self.global_updates % 500 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.save(self.model.state_dict(), f"checkpoint_main_step_{self.global_updates}.pt")
                    if self.use_avg_model:
                        torch.save(self.avg_model.state_dict(), f"checkpoint_avg_step_{self.global_updates}.pt")

                if self.psro_pool and self.psro_refresh_every > 0 and self.global_updates % self.psro_refresh_every == 0:
                    self.refresh_psro_opponent()

                if self.exploiter_enabled and self.global_updates > 0 and self.global_updates % self.exploiter_every == 0 and not self._in_exploiter:
                    self._enter_exploiter_phase(train_count_obj)
                    print(self.q_pivot_logger.sample_per_street(self.action_names))

    @torch.no_grad()
    def _compute_raise_sizes_from_logits(self, last_logits, min_bet_tokens, max_bets, pot_sizes):
        start_id = self.min_size_token_id
        num_sizes = len(self.sizes_floats)
        size_logits = last_logits[:, start_id: start_id + num_sizes]
        offsets = (torch.tensor(min_bet_tokens, device=self.device) - start_id).unsqueeze(1)
        
        size_logits = size_logits.masked_fill(~(torch.arange(num_sizes, device=self.device).unsqueeze(0) >= offsets), float('-inf'))
        chosen_percents = self.torch_sizes_float[torch.multinomial(torch.nan_to_num(torch.softmax(size_logits, dim=1), nan=1e-5), num_samples=1).squeeze(1)]
        bets = (torch.tensor(pot_sizes, device=self.device) * (chosen_percents / 100.0)).long()
        return [int(min(amt, cap) if cap > 0 else amt) for amt, cap in zip(bets.tolist(), max_bets)]

    def process_hand_cpu(self, hand):
        action_space = hand.get_action_space()
        pot_size = hand.pot_size()
        min_bet_token = self.min_size_token_id + min(np.searchsorted(self.sizes_floats, (action_space['min_bet'] / pot_size) * 100, side='left'), len(self.sizes_floats) - 1) if 'min_bet' in action_space else 0
        max_bet = action_space.get('max_bet', 0)
        turn_idx = hand.state.turn_index
        can_raise = 'min_bet' in action_space and action_space['min_bet'] < max_bet
        return (self.encoder.encode(json.dumps(hand.get_u_hand(turn_idx))), min_bet_token, max_bet, pot_size, 'check' in action_space, can_raise, 'call' in action_space, ('call' in action_space and not can_raise), turn_idx)

    def _eval_step_to_terminal(self, items, opp, max_steps=400):
        active = [it for it in items if not it['hand'].done]
        steps = 0
        while active and steps < max_steps:
            hero_turn = [it for it in active if it['hand'].state.turn_index == it['hero']]
            opp_turn  = [it for it in active if it['hand'].state.turn_index != it['hero']]
            if hero_turn:
                hs = [it['hand'] for it in hero_turn]
                acts, szs = self.select_action_batch(hs, temperature=1.0, opp_source='active', explore_eps=0.0)
                for h, a, s in zip(hs, acts, szs):
                    self.apply_action(h, a, s)
            if opp_turn:
                oh = [it['hand'] for it in opp_turn]
                if opp in ('ref', 'ema', 'active'):
                    acts, szs = self.select_action_batch(oh, temperature=1.0, opp_source=opp, explore_eps=0.0)
                    for h, a, s in zip(oh, acts, szs):
                        self.apply_action(h, a, s)
                elif opp == 'calling_station':
                    for h in oh:
                        a, s = self._passive_action(h)
                        self.apply_action(h, a, s)
                elif opp == 'always_fold':
                    for h in oh:
                        a, s = self._foldout_action(h)
                        self.apply_action(h, a, s)
            active = [it for it in active if not it['hand'].done]
            steps += 1
        return items

    def evaluate_winrate(self, n_decks, opp, rake=False, batch_decks=48, max_steps=400):
        n_players = 6
        per_pos = {p: [] for p in range(n_players)}
        deck_means = []
        done = 0
        while done < n_decks:
            b = min(batch_decks, n_decks - done)
            items = []
            for d in range(b):
                seed = Hand()
                seed.shuffle()
                for hero in range(n_players):
                    items.append({'hand': copy.deepcopy(seed), 'hero': hero, 'deck': done + d})
            self._eval_step_to_terminal(items, opp, max_steps=max_steps)
            by_deck = {}
            for it in items:
                h = it['hand']
                if not h.done:
                    continue
                p = float(h.state.payoffs[it['hero']])
                if rake and p > 0:
                    p -= min(p * 0.05, 2.0 * float(h.big_blind))
                v = p / max(float(h.big_blind), 1.0)
                per_pos[it['hero']].append(v)
                by_deck.setdefault(it['deck'], []).append(v)
            for d, vals in by_deck.items():
                if vals:
                    deck_means.append(float(np.mean(vals)))
            done += b

        def stats(xs, n_unit=None):
            a = np.asarray(xs, dtype=np.float64)
            n = len(a)
            if n == 0:
                return (0.0, 0.0, 0)
            return (a.mean() * 100.0, 1.96 * (a.std(ddof=1) / np.sqrt(n)) * 100.0, n_unit if n_unit is not None else n)

        return stats(deck_means, n_unit=sum(len(per_pos[p]) for p in range(n_players))), {p: stats(per_pos[p]) for p in range(n_players)}

    def run_eval_battery(self, n_decks=2000, opponents=('ref', 'calling_station', 'always_fold'), rake=False):
        print(f"\nEvaluating vs {opponents} over {n_decks} decks...")
        results = {}
        for opp in opponents:
            overall, by_pos = self.evaluate_winrate(n_decks, opp, rake=rake)
            results[opp] = (overall, by_pos)
            print(f"Vs {opp:<15} | Winrate: {overall[0]:>+10.3f} bb/100 | 95% CI: {overall[1]:>10.3f}")
        return results


if __name__ == '__main__':
    sim = Simulator()

    if os.environ.get('RUN_EVAL_MODE'):
        hero_path = os.environ.get('EVAL_HERO_CKPT')
        opp_path = os.environ.get('EVAL_OPP_CKPT')
        n_decks = int(os.environ.get('EVAL_DECKS_COUNT', '2000'))
        opponents = ['ref', 'calling_station', 'always_fold']
        if hero_path:
            sim.model.load_state_dict(torch.load(hero_path, map_location=sim.device, weights_only=True))
        if opp_path:
            sim.ema_model.load_state_dict(torch.load(opp_path, map_location=sim.device, weights_only=True))
            opponents.insert(1, 'ema')
        
        threading.Thread(target=sim.inference_server_loop, daemon=True).start()
        threading.Thread(target=sim.inference_server_loop_ema, daemon=True).start()
        threading.Thread(target=sim.inference_server_loop_ref, daemon=True).start()
        time.sleep(2.0)
        sim.run_eval_battery(n_decks=n_decks, opponents=tuple(opponents))
        sys.exit(0)

    experience_queue = queue.Queue(maxsize=10000)
    global_train_count = ThreadSafeCounter()

    threading.Thread(target=sim.inference_server_loop, daemon=True).start()
    threading.Thread(target=sim.inference_server_loop_ema, daemon=True).start()
    threading.Thread(target=sim.inference_server_loop_ref, daemon=True).start()
    threading.Thread(target=sim.inference_server_loop_psro, daemon=True).start()

    num_generators = 4
    for i in range(num_generators):
        threading.Thread(target=sim.worker_loop, args=(i, experience_queue, global_train_count), daemon=True).start()

    try:
        sim.run_trainer(experience_queue, global_train_count)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
