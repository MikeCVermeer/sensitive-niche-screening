"""Inference for the frozen research heads with explicitly projected evidence.

The research feature projectors are not yet distributable; callers must supply
identically projected features. Scores cannot be reproduced from vectors alone.
"""
from pathlib import Path
import json
import numpy as np


def _sigmoid(x, limit=32):
    return 1 / (1 + np.exp(-np.clip(x, -limit, limit)))


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _features(vectors, features, width):
    vectors = np.asarray(vectors, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 1024 or not len(vectors):
        raise ValueError('vectors must have shape (pages, 1024), with at least one page')
    if features.shape != (len(vectors), 9, width):
        raise ValueError(f'features must have shape (pages, 9, {width})')
    if not np.isfinite(vectors).all() or not np.isfinite(features).all():
        raise ValueError('inputs must be finite')
    if np.any(~np.any(vectors, axis=1)):
        raise ValueError('missing vectors cannot be scored as negative')
    return vectors, features


class FrozenScreen:
    def __init__(self, directory):
        directory = Path(directory)
        self.config = json.loads((directory / 'config.json').read_text())
        with np.load(directory / 'screening-v1.npz', allow_pickle=False) as f:
            self.arrays = {k: f[k] for k in f.files}

    def _semantics(self, v):
        a = self.arrays
        topic = v @ a['topic'].T
        intent = v @ a['intent'].T
        own = (v @ a['prototype'].reshape(27, 1024).T).reshape(-1, 9, 3)
        return np.concatenate((topic[:, :, None]*8,
            np.broadcast_to(intent[:, None, :]*8, own.shape),
            (own-own.mean(axis=2, keepdims=True))*15,
            topic[:, :, None]*intent[:, None, :]*40), axis=2)

    def score_pages(self, vectors, a7_features):
        v, x = _features(vectors, a7_features, 19)
        a = self.arrays
        pooled = np.concatenate((x, self._semantics(v)), axis=2)
        raw = _sigmoid(((pooled-a['mean'])/a['scale']) @ a['weights'] + a['bias'])
        cfg = self.config['a7']
        def calibrate(score, name):
            slope, intercept = cfg['calibration'][name]
            logit = np.log(np.clip(score, 1e-6, 1-1e-6)/np.clip(1-score, 1e-6, 1))
            return _sigmoid(slope*logit+intercept, 30)
        alarm = calibrate(raw[:, :, 1:].max(axis=2), 'alarm')
        any_score = np.maximum(alarm, calibrate(raw.max(axis=2), 'any'))
        return {'raw_relations': raw, 'commercial_score': alarm,
                'commercial': alarm >= cfg['threshold']['alarm'], 'any_score': any_score,
                'any': any_score >= cfg['threshold']['any']}

    def operator_projection(self, vectors, operator_features, homepage):
        v, x = _features(vectors, operator_features, 16)
        homepage = np.asarray(homepage, dtype=bool)
        if homepage.shape != (len(v),):
            raise ValueError('homepage must have shape (pages,)')
        sem = self._semantics(v)[:, :, 0]
        vec = _unit((v*(1+3*homepage)[:, None]).sum(axis=0))
        topic = x[:, :, :6].max(axis=2)
        offer = x[:, :, [9, 13, 14, 15]].max(axis=2)
        named = x[:, :, [0, 1, 2, 4]].max(axis=2)
        def home_max(values):
            return values[homepage].max(axis=0) if homepage.any() else np.zeros(9)
        info = np.stack([sem.mean(0),sem.max(0),np.quantile(sem,.75,axis=0),home_max(sem),
            topic.astype(bool).mean(0),named.mean(0),np.minimum(topic,1).mean(0)*offer.mean(0),
            home_max(named),home_max(offer),x[:,:,12].mean(0),np.quantile(x[:,:,5],.75,axis=0),
            np.minimum(topic,1).mean(0)*sem.mean(0)],axis=1).astype('f')
        pooled = np.concatenate((info, self._semantics(vec[None,:])[0]),axis=1)
        return pooled

    def score_site(self, vectors, a7_features, operator_features, homepage):
        pages = self.score_pages(vectors, a7_features)
        pooled = self.operator_projection(vectors, operator_features, homepage)
        op = _sigmoid(pooled @ self.arrays['operator_weights'] + self.arrays['operator_bias'],35)[:,0]
        return {'topics': self.config['topics'], 'commercial_score': pages['commercial_score'].max(axis=0),
                'commercial': pages['commercial'].any(axis=0), 'operator_score': op,
                'operator': op >= self.config['operator_threshold']}


def load_model(path=None, repo_id='MikeCVermeer/sensitive-niche-screening'):
    """Load a local weights directory, or download the two files from Hugging Face."""
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, allow_patterns=['config.json', 'screening-v1.npz'])
    return FrozenScreen(path)


def score_pages(model, vectors, a7_features):
    return model.score_pages(vectors, a7_features)


def score_site(model, vectors, a7_features, operator_features, homepage):
    return model.score_site(vectors, a7_features, operator_features, homepage)
