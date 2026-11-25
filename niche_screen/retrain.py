"""Refit the two frozen architectures on pre-projected, site-split data.

Input is an NPZ with numeric arrays, documented in docs/BRING_YOUR_OWN.md.
No original examples, cue lists or external services are needed.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from .inference import load_model
from .evaluation import f1


def _binary(x, y):
    if len(np.unique(y)) != 2:
        raise ValueError('each fit requires both positive and negative assessed labels')
    n = int(y.sum())
    m = LogisticRegression(C=1, solver='lbfgs', max_iter=1000,
                           class_weight={0: 1., 1: min(16., np.sqrt((len(y)-n)/n))})
    m.fit(x, y)
    return m.coef_[0].astype('f'), np.float32(m.intercept_[0])


def _choose(scores, labels):
    if not np.any(labels):
        raise ValueError('development data must contain positives')
    candidates = np.unique(scores)
    return float(max(candidates, key=lambda t: (f1(labels, scores >= t), t)))


def _platt(scores, labels):
    if len(np.unique(labels)) != 2:
        raise ValueError('development calibration needs positive and negative labels')
    x = np.log(np.clip(scores, 1e-6, 1-1e-6)/np.clip(1-scores,1e-6,1))[:,None]
    m = LogisticRegression(C=1,max_iter=1000).fit(x,labels)
    if m.coef_[0,0] < 0:
        p = np.mean(labels)
        return [0., float(np.log(p/(1-p)))]
    return [float(m.coef_[0,0]),float(m.intercept_[0])]


def fit(data, output, base):
    model = load_model(base)
    v = np.asarray(data['vectors'], dtype='f')
    x = np.asarray(data['page_features'], dtype='f')
    ox = np.asarray(data['operator_features'], dtype='f')
    home = np.asarray(data['homepage'], dtype=bool)
    site = np.asarray(data['site_index'], dtype=int)
    splits = np.asarray(data['site_split'], dtype=int)
    labels = np.asarray(data['page_labels'], dtype=int)
    ol = np.asarray(data['operator_labels'], dtype=int)
    nsites = len(splits)
    if (labels.shape != (len(v),9) or ol.shape != (nsites,9) or home.shape != (len(v),)
            or site.shape != (len(v),) or np.any(site < 0) or np.any(site >= nsites)
            or not np.isin(splits, [0,1,2]).all() or not np.isin(labels,[-1,0,1,2,3]).all()
            or not np.isin(ol,[-1,0,1]).all() or set(site) != set(range(nsites))):
        raise ValueError('invalid shapes, codes, or site alignment')
    if set(splits) != {0,1,2}:
        raise ValueError('site_split must include train, development and test')
    from .inference import _features
    _features(v,x,19); _features(v,ox,16)
    z = np.concatenate((x,model._semantics(v)),axis=2)
    train = splits[site] == 0
    # Ignore unassessed page-topic pairs, rather than treating them as negatives.
    known = train[:,None] & (labels >= 0)
    a = z[known]
    model.arrays['mean'] = a.mean(0)
    model.arrays['scale'] = np.maximum(a.std(0),.25)
    a = (a-model.arrays['mean'])/model.arrays['scale']
    heads = [_binary(a,(labels[known] == n).astype(int)) for n in (1,2,3)]
    model.arrays['weights'] = np.stack([w for w,b in heads],axis=1)
    model.arrays['bias'] = np.array([b for w,b in heads],dtype='f')
    dev = (splits[site] == 1)[:,None] & (labels >= 0)
    raw = model.score_pages(v,x)['raw_relations']
    alarm_raw = raw[:,:,1:].max(2)
    any_raw = raw.max(2)
    cal = {'alarm':_platt(alarm_raw[dev],(labels[dev]>=2).astype(int)),
           'any':_platt(any_raw[dev],(labels[dev]>0).astype(int))}
    model.config['a7']['calibration'] = cal
    score = model.score_pages(v,x)
    model.config['a7']['threshold'] = {'alarm':_choose(score['commercial_score'][dev],labels[dev]>=2),
                                       'any':_choose(score['any_score'][dev],labels[dev]>0)}
    projections = np.stack([model.operator_projection(v[site==i],ox[site==i],home[site==i]) for i in range(nsites)])
    known_op = (splits==0)[:,None] & (ol >= 0)
    opx = projections[known_op]; opy = ol[known_op]
    mean = opx.mean(0); scale = np.maximum(opx.std(0),.20)
    w,b = _binary((opx-mean)/scale,opy)
    model.arrays['operator_weights'] = (w/scale)[:,None]
    model.arrays['operator_bias'] = np.array([b-mean@(w/scale)],dtype='f')
    from .inference import _sigmoid
    oscore = _sigmoid(projections@model.arrays['operator_weights']+model.arrays['operator_bias'],35)[:,:,0]
    dev_op = (splits==1)[:,None] & (ol>=0)
    model.config['operator_threshold'] = _choose(oscore[dev_op],ol[dev_op])
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'screening-v1.npz',**model.arrays)
    (output/'config.json').write_text(json.dumps(model.config,indent=2)+'\n')
    from .evaluation import site_bootstrap
    rows=[]
    for s in np.flatnonzero(splits == 2):
        ix = site==s
        pred = model.score_site(v[ix],x[ix],ox[ix],home[ix])
        for n in range(9):
            y=labels[ix,n]
            if (y>=2).any(): truth=1
            elif (y<0).any(): truth=None
            else: truth=0
            rows.append({'site':int(s),'label':truth,'prediction':int(pred['commercial'][n])})
    return site_bootstrap(rows,200)


def synthetic():
    rng=np.random.default_rng(9); sites=60; per=3; n=sites*per
    v=rng.normal(size=(n,1024)).astype('f');v/=np.linalg.norm(v,axis=1,keepdims=True)
    x=rng.uniform(size=(n,9,19)).astype('f');ox=rng.uniform(size=(n,9,16)).astype('f')
    labels=rng.choice([0,1,2,3],size=(n,9),p=[.55,.15,.20,.10]).astype('i1')
    x[:,:,0] += labels>=2
    op=rng.integers(0,2,size=(sites,9),dtype='i1')
    return dict(vectors=v,page_features=x,operator_features=ox,homepage=np.tile([1,0,0],sites),
        site_index=np.repeat(np.arange(sites),per),site_split=np.repeat([0,1,2],[40,10,10]),
        page_labels=labels,operator_labels=op)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', help='NPZ data with site-disjoint split codes')
    ap.add_argument('--output',help='directory for trained NPZ and JSON')
    ap.add_argument('--base',default=str(Path(__file__).resolve().parents[1]/'weights'))
    ap.add_argument('--demo',action='store_true',help='fit and evaluate fictional data in a temporary directory')
    args=ap.parse_args()
    if args.demo:
        with tempfile.TemporaryDirectory() as out:
            print(json.dumps({'data':'fictional synthetic features; scores do not measure real-world accuracy',
                              'held_out_site_alarm':fit(synthetic(),out,args.base)},indent=2))
    elif args.input and args.output:
        with np.load(args.input,allow_pickle=False) as data:
            print(json.dumps({'held_out_site_alarm':fit(data,args.output,args.base)},indent=2))
    else:
        ap.error('use --demo, or both --input and --output')

if __name__=='__main__': main()
