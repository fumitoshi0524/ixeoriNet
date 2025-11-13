package nn

import (
    "errors"
    "fmt"
    "math"

    "github.com/fumitoshi0524/ixeoriNet/tensor"
)

// Attention implements a Transformer-style (scaled dot-product) attention block.
// It computes Q/K/V projections, scaled dot-product attention, projects output
// back and applies a residual connection plus LayerNorm. The returned tensor
// is pooled across the sequence (mean) to shape [batch, dim] to be compatible
// with existing classifier heads in examples.
type Attention struct {
    seq int
    dim int
    // projections: weights shaped [out, in]
    wq *tensor.Tensor
    wk *tensor.Tensor
    wv *tensor.Tensor
    wo *tensor.Tensor
    ln *LayerNorm
}

// NewAttention creates a Transformer-style attention block for inputs with
// sequence length seq and embedding dimension dim.
func NewAttention(seq, dim int) *Attention {
    // init weights [dim, dim]
    wq := tensor.Randn(dim, dim)
    wk := tensor.Randn(dim, dim)
    wv := tensor.Randn(dim, dim)
    wo := tensor.Randn(dim, dim)
    // small scaling init
    scale := math.Sqrt(2.0 / float64(dim+dim))
    wq.Scale(scale)
    wk.Scale(scale)
    wv.Scale(scale)
    wo.Scale(scale)
    wq.SetRequiresGrad(true)
    wk.SetRequiresGrad(true)
    wv.SetRequiresGrad(true)
    wo.SetRequiresGrad(true)
    ln := NewLayerNorm([]int{dim}, 1e-5, true)
    return &Attention{seq: seq, dim: dim, wq: wq, wk: wk, wv: wv, wo: wo, ln: ln}
}

func (a *Attention) Parameters() []*tensor.Tensor {
    params := []*tensor.Tensor{a.wq, a.wk, a.wv, a.wo}
    params = append(params, a.ln.Parameters()...)
    return params
}

func (a *Attention) ZeroGrad() {
    for _, p := range a.Parameters() {
        if p != nil {
            p.ZeroGrad()
        }
    }
}

func (a *Attention) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
    if input == nil {
        return nil, errors.New("attention: nil input")
    }
    shape := input.Shape()
    if len(shape) != 3 {
        return nil, errors.New("attention expects rank-3 input [batch, seq, dim]")
    }
    batch, seq, dim := shape[0], shape[1], shape[2]
    if seq != a.seq || dim != a.dim {
        return nil, fmt.Errorf("attention input shape mismatch: got seq=%d dim=%d want seq=%d dim=%d", seq, dim, a.seq, a.dim)
    }

    // flatten tokens to 2D for projection
    x2d, err := input.Reshape(batch*seq, dim)
    if err != nil {
        return nil, err
    }
    q2d, err := tensor.MatMul(x2d, a.wq.MustTranspose())
    if err != nil {
        return nil, err
    }
    k2d, err := tensor.MatMul(x2d, a.wk.MustTranspose())
    if err != nil {
        return nil, err
    }
    v2d, err := tensor.MatMul(x2d, a.wv.MustTranspose())
    if err != nil {
        return nil, err
    }

    // will store pooled outputs
    pooledList := make([]*tensor.Tensor, 0, batch)

    for b := 0; b < batch; b++ {
        rowOffset := b * seq
        Q, err := tensor.SliceRows2D(q2d, rowOffset, seq)
        if err != nil {
            return nil, err
        }
        K, err := tensor.SliceRows2D(k2d, rowOffset, seq)
        if err != nil {
            return nil, err
        }
        V, err := tensor.SliceRows2D(v2d, rowOffset, seq)
        if err != nil {
            return nil, err
        }
        // scores = Q @ K.T  -> (seq, seq)
        Kt := K.MustTranspose()
        scores, err := tensor.MatMul(Q, Kt)
        if err != nil {
            return nil, err
        }
        // scale (use a proper tensor op to preserve autograd)
        scaleVal := 1.0 / math.Sqrt(float64(dim))
        scaleT := tensor.Full(scaleVal, scores.Shape()...)
        scoresScaled, err := tensor.Mul(scores, scaleT)
        if err != nil {
            return nil, err
        }
        // softmax
        probs, err := tensor.Softmax(scoresScaled, 1)
        if err != nil {
            return nil, err
        }
        // context = probs @ V  -> (seq, dim)
        ctx, err := tensor.MatMul(probs, V)
        if err != nil {
            return nil, err
        }
        // project back
        outProj2d, err := tensor.MatMul(ctx, a.wo.MustTranspose())
        if err != nil {
            return nil, err
        }

        // residual: original tokens for this batch
        Xorig, err := tensor.SliceRows2D(x2d, rowOffset, seq)
        if err != nil {
            return nil, err
        }
        // add residual: out = Xorig + outProj2d
        outTok, err := tensor.Add(Xorig, outProj2d)
        if err != nil {
            return nil, err
        }
        // layernorm over last dim
        outNorm, err := a.ln.Forward(outTok)
        if err != nil {
            return nil, err
        }

        // pool across sequence (mean over rows) to produce [dim]
        sumVec, err := tensor.SumAxis(outNorm, 0)
        if err != nil {
            return nil, err
        }
        denom := tensor.Full(float64(seq), sumVec.Shape()...)
        meanVec, err := tensor.Div(sumVec, denom)
        if err != nil {
            return nil, err
        }
        pooledList = append(pooledList, meanVec)
    }

    // stack pooled vectors to shape [batch, dim]
    out, err := tensor.Stack(0, pooledList...)
    if err != nil {
        return nil, err
    }
    return out, nil
}
