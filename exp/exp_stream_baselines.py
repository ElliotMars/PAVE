import os
import random
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from exp.exp_basic import Exp_Basic
from models.dsof import DSOFStudent
from models.er import ERForecaster
from models.fsnet import FSNetForecaster
from models.onenet import OneNetForecaster
from models.proceed import ProceedForecaster
from models.patchtst_dgrad import PatchTSTDGrad
from utils.metrics import cumavg, metric
from utils.progressive_baseline_feedback import (
    ProgressiveBaselineDiagnostics,
    ProgressiveBaselineFeedbackManager,
    ProgressiveBaselineRecord,
    progressive_partial_mse,
)
from utils.tools import EarlyStopping, adjust_learning_rate


class ExpStreamBaseline(Exp_Basic):
    method = None

    def __init__(self, args):
        self.args, self.online = args, args.online_learning
        self.device = self._acquire_device()
        if self.method == "fsnet":
            self.model = FSNetForecaster(args, self.device)
        elif self.method == "onenet":
            self.model = OneNetForecaster(args, self.device)
        elif self.method == "proceed":
            self.model = ProceedForecaster(args)
        else:
            self.model = ERForecaster(args) if self.method == "er" else PatchTSTDGrad(args)
        self.model = self.model.to(self.device)
        self.student = DSOFStudent(args).to(self.device) if self.method == "dsof" else None
        self.buffer = deque(maxlen=args.replay_buffer_size)
        self.opt = self.opt_student = None
        self.prev_student = None
        self.progressive_baseline_fb = bool(
            getattr(args, "progressive_baseline_fb", False)
        )
        self.progressive_protocol_diagnostics = None

    def _get_data(self, flag):
        mapping = defaultdict(lambda: Dataset_Custom, {
            "ETTh1": Dataset_ETT_hour, "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute, "ETTm2": Dataset_ETT_minute,
        })
        ds = mapping[self.args.data](root_path=self.args.root_path, data_path=self.args.data_path,
            flag=flag, delay_fb=self.args.delay_fb,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features, target=self.args.target, inverse=self.args.inverse,
            timeenc=2, freq=self.args.freq, cols=self.args.cols)
        bsz = self.args.test_bsz if flag == "test" else self.args.batch_size
        loader = DataLoader(ds, batch_size=bsz, shuffle=flag == "train",
            num_workers=self.args.num_workers, drop_last=flag == "train")
        print(flag, len(ds)); return ds, loader

    def _target(self, y):
        f_dim = -1 if self.args.features == "MS" else 0
        return y.float().to(self.device)[:, -self.args.pred_len:, f_dim:]

    def _forward(self, x, mark=None, offline=False):
        if self.method in ("fsnet", "onenet"):
            return self.model(x, mark)
        if self.method == "proceed":
            return self.model(x, mark, backbone_only=offline)
        if self.method == "dsof":
            return self.model(x) if offline else self.model(x) + self.student(x)
        return self.model(x)

    def _store_grad(self):
        if hasattr(self.model, "store_grad"):
            self.model.store_grad()

    @contextmanager
    def _fsnet_state_updates(self, enabled):
        modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "state_updates_enabled")
        ]
        previous = [module.state_updates_enabled for module in modules]
        try:
            for module in modules:
                module.state_updates_enabled = enabled
            yield
        finally:
            for module, old_value in zip(modules, previous):
                module.state_updates_enabled = old_value

    def _current_observation(self, x):
        f_dim = -1 if self.args.features == "MS" else 0
        observation = x.float()[:, -1, f_dim:].reshape(-1)
        if tuple(observation.shape) != (self.args.c_out,):
            raise ValueError(
                "observable target has shape {}, expected {}".format(
                    tuple(observation.shape), (self.args.c_out,)
                )
            )
        return observation

    def train(self, setting):
        _, train_loader = self._get_data("train"); _, val_loader = self._get_data("val")
        path = os.path.join(self.args.checkpoints, setting); os.makedirs(path, exist_ok=True)
        params = self.model.backbone.parameters() if self.method == "proceed" else self.model.parameters()
        self.opt = optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion, stopper = nn.MSELoss(), EarlyStopping(self.args.patience, verbose=True)
        for epoch in range(self.args.train_epochs):
            self.model.train(); losses = []
            for x, y, mark, _ in train_loader:
                x, mark, y = x.float().to(self.device), mark.float().to(self.device), self._target(y)
                self.opt.zero_grad(); pred = self._forward(x, mark, offline=True)
                loss = criterion(pred, y); loss.backward(); self.opt.step(); self._store_grad()
                losses.append(loss.item())
            val_loss = self.vali(None, val_loader, criterion)
            print("Epoch: {} | Train Loss: {:.6f} Vali Loss: {:.6f}".format(epoch+1, np.mean(losses), val_loss))
            stopper(val_loss, self.model, path)
            if stopper.early_stop: break
            adjust_learning_rate(self.opt, epoch + 1, self.args)
        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location=self.device))
        return self.model

    def load_pretrained(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)); return self.model

    def vali(self, data, loader, criterion):
        self.model.eval(); losses=[]
        with torch.no_grad():
            for x,y,mark,_ in loader:
                pred=self._forward(x.float().to(self.device), mark.float().to(self.device), offline=True)
                losses.append(criterion(pred,self._target(y)).item())
        return float(np.mean(losses))

    def _sample_replay(self):
        if not self.buffer: return None
        items = random.sample(list(self.buffer), min(len(self.buffer), self.args.replay_batch_size))
        return torch.cat([z[0] for z in items]), torch.cat([z[1] for z in items]), torch.cat([z[2] for z in items])

    def _online_update(self, x, y, mark):
        criterion = nn.MSELoss()
        if self.method == "dsof":
            self.opt_student.zero_grad()
            with torch.no_grad(): teacher = self.model(x)
            student_pred = self.student(x)
            loss = criterion(student_pred, y - teacher)
            if self.prev_student is not None and self.args.pred_len > 1:
                loss = loss + self.args.dsof_td_weight * criterion(self.prev_student[:,1:], student_pred[:,:-1])
            loss.backward(); self.opt_student.step(); self.prev_student = student_pred.detach()
            replay = self._sample_replay()
            if replay:
                rx, ry, _ = replay; self.opt.zero_grad(); criterion(self.model(rx), ry).backward(); self.opt.step()
        elif self.method == "proceed":
            self.opt.zero_grad(); loss=criterion(self.model(x, mark), y); loss.backward(); self.opt.step()
            self.model.observe(x,y)
        else:
            self.opt.zero_grad(); loss=criterion(self._forward(x,mark),y)
            if self.method == "er":
                replay=self._sample_replay()
                if replay:
                    rx,ry,rm=replay; loss=loss+criterion(self.model(rx),ry)
            loss.backward(); self.opt.step(); self._store_grad()
        self.buffer.append((x.detach().clone(),y.detach().clone(),mark.detach().clone()))

    def _progressive_feedback_update(self, events):
        if not events:
            return None
        historical_x = torch.cat(
            [event.record.x for event in events], dim=0
        ).float().to(self.device)
        historical_mark = torch.cat(
            [event.record.x_mark for event in events], dim=0
        ).float().to(self.device)
        horizon_indices = torch.tensor(
            [event.horizon_index for event in events],
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.stack(
            [event.target for event in events], dim=0
        ).float().to(self.device)

        self.model.train()
        self.opt.zero_grad()
        # Historical replay reads the FSNet fast/slow state but must not advance
        # it once per pending record. The current-origin prediction below remains
        # the sole state-mutating forward, matching the native stream semantics.
        with self._fsnet_state_updates(False):
            predictions = self._forward(historical_x, historical_mark)
            loss = progressive_partial_mse(
                predictions, horizon_indices, targets
            )
        loss.backward()
        self.opt.step()
        self._store_grad()
        self.progressive_protocol_diagnostics.optimizer_step()
        return loss.detach()

    def _progressive_online_batch(self, manager, x, y, mark):
        batch_preds, batch_trues = [], []
        for i in range(x.shape[0]):
            origin = self.progressive_protocol_diagnostics.total_origins
            current_x = x[i : i + 1]
            current_mark = mark[i : i + 1]
            observation = self._current_observation(current_x)
            events = manager.release(origin, observation)
            self.progressive_protocol_diagnostics.begin_origin(origin, events)
            self._progressive_feedback_update(events)

            self.model.eval()
            with torch.no_grad():
                prediction = self._forward(current_x, current_mark)
            record = ProgressiveBaselineRecord(
                origin=origin,
                x=current_x,
                x_mark=current_mark,
                method=self.method,
            )
            manager.add_record(record)
            self.progressive_protocol_diagnostics.prediction()
            batch_preds.append(prediction)
            # Full forecast truth is evaluation-only and is never retained by
            # the manager or ProgressiveBaselineRecord.
            batch_trues.append(y[i : i + 1])

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def _delayed_online_batch(self, feedback_queue, x, y, mark):
        batch_preds, batch_trues = [], []
        for i in range(x.shape[0]):
            if len(feedback_queue) >= self.args.pred_len:
                released_x, released_y, released_mark = feedback_queue.popleft()
                self.model.train()
                if self.student is not None:
                    self.student.train()
                self._online_update(released_x, released_y, released_mark)

            current = (
                x[i : i + 1].detach().clone(),
                y[i : i + 1].detach().clone(),
                mark[i : i + 1].detach().clone(),
            )
            self.model.eval()
            if self.student is not None:
                self.student.eval()
            with torch.no_grad():
                pred = self._forward(current[0], current[2])
            feedback_queue.append(current)
            batch_preds.append(pred)
            batch_trues.append(current[1])

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def test(self, setting):
        _, loader=self._get_data("test")
        if self.method == "proceed":
            self.model.freeze_backbone(True); self.opt=optim.Adam(self.model.adapter_parameters(),lr=self.args.proceed_online_lr)
        else:
            self.opt=optim.AdamW(self.model.parameters(),lr=self.args.baseline_online_lr,weight_decay=self.args.weight_decay)
        if self.method == "dsof":
            self.opt_student=optim.Adam(self.student.parameters(),lr=self.args.dsof_student_lr)
        self.buffer.clear(); preds=[]; trues=[]; records=[[],[],[],[],[]]; start=time.time()
        progressive_manager = None
        if self.progressive_baseline_fb and self.online != "none":
            progressive_manager = ProgressiveBaselineFeedbackManager(
                pred_len=self.args.pred_len, c_out=self.args.c_out
            )
            self.progressive_protocol_diagnostics = (
                ProgressiveBaselineDiagnostics(self.args.pred_len)
            )
            print(
                "[PROGRESSIVE_BASELINE_FB] release -> learn -> predict -> store"
            )
        else:
            self.progressive_protocol_diagnostics = None
        feedback_queue = (
            deque()
            if progressive_manager is None
            and self.args.delay_fb
            and self.online != "none"
            else None
        )
        if feedback_queue is not None:
            print("[DELAY_FB] rolling origins; feedback delay={} steps".format(self.args.pred_len))
        processed_origins = 0
        for x,y,mark,_ in tqdm(loader):
            max_steps = int(getattr(self.args, "max_online_steps", -1))
            if max_steps > 0:
                remaining = max_steps - processed_origins
                if remaining <= 0:
                    break
                if x.shape[0] > remaining:
                    x, y, mark = x[:remaining], y[:remaining], mark[:remaining]
            x,mark,y=x.float().to(self.device),mark.float().to(self.device),self._target(y)
            if progressive_manager is not None:
                pred, y = self._progressive_online_batch(
                    progressive_manager, x, y, mark
                )
            elif feedback_queue is not None:
                pred, y = self._delayed_online_batch(feedback_queue, x, y, mark)
            else:
                self.model.eval()
                if self.student is not None:self.student.eval()
                with torch.no_grad(): pred=self._forward(x,mark)
                if self.online != "none":
                    self.model.train()
                    if self.student is not None:self.student.train()
                    self._online_update(x,y,mark)
            processed_origins += int(pred.shape[0])
            pf=rearrange(pred,"b t d -> b (t d)"); tf=rearrange(y,"b t d -> b (t d)")
            preds.append(pf.cpu());trues.append(tf.cpu())
            for arr,val in zip(records,metric(pf.cpu().numpy(),tf.cpu().numpy())):arr.append(val)
        if progressive_manager is not None:
            self.progressive_protocol_diagnostics.finish(len(progressive_manager))
        curves=[cumavg(x) for x in records]; vals=[x[-1] for x in curves]; elapsed=time.time()-start
        print("mse:{}, mae:{}, time:{}".format(vals[1],vals[0],elapsed))
        return vals+[elapsed],curves[0],curves[1],torch.cat(preds).numpy(),torch.cat(trues).numpy()

    def save_progressive_baseline_diagnostics(self, result_directory):
        if self.progressive_protocol_diagnostics is None:
            raise RuntimeError("progressive baseline diagnostics are unavailable")
        return self.progressive_protocol_diagnostics.save(result_directory)


class ExpER(ExpStreamBaseline): method="er"
class ExpFSNet(ExpStreamBaseline): method="fsnet"
class ExpOneNet(ExpStreamBaseline): method="onenet"
class ExpDSOF(ExpStreamBaseline): method="dsof"
class ExpProceed(ExpStreamBaseline): method="proceed"
