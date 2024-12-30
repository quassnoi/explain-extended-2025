CREATE OR REPLACE FUNCTION EMB(_step INT)
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(COS(value)) || ARRAY_AGG(SIN(value)) AS t_emb
FROM    GENERATE_SERIES(0, 31) AS dim
CROSS JOIN LATERAL
        (
        SELECT  _step * EXP(-LN(10000) * dim / 32.0) AS value
        ) q;

$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION CONV2D3X3(_input REAL[][][], _weight REAL[][][][], _bias REAL[], _stride INT = 1)
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(value)
FROM    (
        SELECT  ARRAY_LENGTH(_input, 3) AS image_width,
                ARRAY_LENGTH(_input, 2) AS image_height
        ) q1
CROSS JOIN LATERAL
        GENERATE_SERIES(0, ARRAY_LENGTH(_weight, 1) - 1) AS c_out
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(value) AS value
        FROM    GENERATE_SERIES(0, image_height - 3 + 2, _stride) ky
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(value) AS value
                FROM    GENERATE_SERIES(0, image_width - 3 + 2, _stride) kx
                CROSS JOIN LATERAL
                        (
                        SELECT  SUM(
                                        COALESCE(_input[c_in + 1][ky - 1 + 1][kx - 1 + 1], 0) * _weight[c_out + 1][c_in + 1][1][1] +
                                        COALESCE(_input[c_in + 1][ky - 1 + 1][kx     + 1], 0) * _weight[c_out + 1][c_in + 1][1][2] +
                                        COALESCE(_input[c_in + 1][ky - 1 + 1][kx + 1 + 1], 0) * _weight[c_out + 1][c_in + 1][1][3] +
                                        COALESCE(_input[c_in + 1][ky     + 1][kx - 1 + 1], 0) * _weight[c_out + 1][c_in + 1][2][1] +
                                        COALESCE(_input[c_in + 1][ky     + 1][kx     + 1], 0) * _weight[c_out + 1][c_in + 1][2][2] +
                                        COALESCE(_input[c_in + 1][ky     + 1][kx + 1 + 1], 0) * _weight[c_out + 1][c_in + 1][2][3] +
                                        COALESCE(_input[c_in + 1][ky + 1 + 1][kx - 1 + 1], 0) * _weight[c_out + 1][c_in + 1][3][1] +
                                        COALESCE(_input[c_in + 1][ky + 1 + 1][kx     + 1], 0) * _weight[c_out + 1][c_in + 1][3][2] +
                                        COALESCE(_input[c_in + 1][ky + 1 + 1][kx + 1 + 1], 0) * _weight[c_out + 1][c_in + 1][3][3] +
                                        0
                                )
                                + _bias[c_out + 1]
                                AS value
                        FROM    GENERATE_SERIES(0, ARRAY_LENGTH(_weight, 2) - 1) AS c_in
                        ) q
                ) q
        ) q2
$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION CONV2D1X1(_input REAL[][][], _weight REAL[][][][], _bias REAL[], _stride INT = 1)
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(value)
FROM    (
        SELECT  ARRAY_LENGTH(_input, 3) AS image_width,
                ARRAY_LENGTH(_input, 2) AS image_height
        ) q1
CROSS JOIN LATERAL
        GENERATE_SERIES(0, ARRAY_LENGTH(_weight, 1) - 1) AS c_out
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(value) AS value
        FROM    GENERATE_SERIES(0, image_height - 1, _stride) ky
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(value) AS value
                FROM    GENERATE_SERIES(0, image_width - 1, _stride) kx
                CROSS JOIN LATERAL
                        (
                        SELECT  SUM(
                                        COALESCE(_input[c_in + 1][ky + 1][kx + 1], 0) * _weight[c_out + 1][c_in + 1][1][1]
                                )
                                + _bias[c_out + 1]
                                AS value
                        FROM    GENERATE_SERIES(0, ARRAY_LENGTH(_weight, 2) - 1) AS c_in
                        ) q
                ) q
        ) q2
$$

LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION LINEAR_1_2(_input REAL[], _weight REAL[][], _bias REAL[])
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(value)
FROM    GENERATE_SUBSCRIPTS(_weight, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  SUM(_input[l2] * _weight[l1][l2]) + _bias[l1] AS value
        FROM    GENERATE_SUBSCRIPTS(_weight, 2) l2
        ) q

$$

LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION LINEAR_2_2(_input REAL[][], _weight REAL[][], _bias REAL[])
RETURNS REAL[][]
AS
$$

SELECT  ARRAY_AGG(value)
FROM    GENERATE_SUBSCRIPTS(_input, 1) i1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(value) AS value
        FROM    GENERATE_SUBSCRIPTS(_weight, 1) w1
        CROSS JOIN LATERAL
                (
                SELECT  SUM(_input[i1][l2] * _weight[w1][l2]) + _bias[w1] AS value
                FROM    GENERATE_SUBSCRIPTS(_weight, 2) l2
                )
        ) q

$$

LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;


CREATE OR REPLACE FUNCTION PLUS_3_1(_a REAL[][][], _b REAL[])
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(v2)
FROM    GENERATE_SUBSCRIPTS(_a, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v3) AS v2
        FROM    GENERATE_SUBSCRIPTS(_a, 2) l2
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(_a[l1][l2][l3] + _b[l1]) AS v3
                FROM    GENERATE_SUBSCRIPTS(_a, 2) l3
                )
        ) q

$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION PLUS_3_2(_a REAL[][][], _b REAL[][])
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(v2)
FROM    GENERATE_SUBSCRIPTS(_a, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v3) AS v2
        FROM    GENERATE_SUBSCRIPTS(_a, 2) l2
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(_a[l1][l2][l3] + _b[l1][l2]) AS v3
                FROM    GENERATE_SUBSCRIPTS(_a, 2) l3
                )
        ) q

$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION PLUS_3_3(_a REAL[][][], _b REAL[][])
RETURNS REAL[]
AS
$$

SELECT  ARRAY_AGG(v2)
FROM    GENERATE_SUBSCRIPTS(_a, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v3) AS v2
        FROM    GENERATE_SUBSCRIPTS(_a, 2) l2
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(_a[l1][l2][l3] + _b[l1][l2][l3]) AS v3
                FROM    GENERATE_SUBSCRIPTS(_a, 2) l3
                )
        ) q

$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION SILU(_v REAL[])
RETURNS REAL[]
AS
$$
SELECT  ARRAY_AGG(v * EXP(v) / (1 + EXP(v)))
FROM    UNNEST(_v) AS v
$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION SILU2(_v REAL[][])
RETURNS REAL[][]
AS
$$
SELECT  ARRAY_AGG(v2)
FROM    GENERATE_SUBSCRIPTS(_v, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v * EXP(v) / (1 + EXP(v))) v2
        FROM    GENERATE_SUBSCRIPTS(_v, 1) l2
        CROSS JOIN LATERAL
                (
                SELECT  _v[l1][l2] AS v
                ) q
        ) q
$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION SILU3(_v REAL[][][])
RETURNS REAL[][][]
AS
$$
SELECT  ARRAY_AGG(v2)
FROM    GENERATE_SUBSCRIPTS(_v, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v3) AS v2
        FROM    GENERATE_SUBSCRIPTS(_v, 2) l2
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v * EXP(v) / (1 + EXP(v))) v3
                FROM    GENERATE_SUBSCRIPTS(_v, 3) l3
                CROSS JOIN LATERAL
                        (
                        SELECT  _v[l1][l2][l3] AS v
                        ) q
                ) q
        ) q
$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

        CREATE OR REPLACE FUNCTION GROUP_NORM(_image REAL[][][], _weight REAL[], _bias REAL[], _groups INT)
        RETURNS REAL[][]
        AS
        $$

        SELECT  value
        FROM    (
                SELECT  ARRAY_LENGTH(_image, 1) / _groups AS group_size
                ) c
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(var_denominator) AS var_denominator,
                        ARRAY_AGG(mean) AS mean
                FROM    GENERATE_SERIES(0, ARRAY_LENGTH(_image, 1) -1, group_size) AS group_number
                CROSS JOIN LATERAL
                        (
                        SELECT  SQRT(VAR_SAMP(value) + 1e-05) AS var_denominator,
                                AVG(value) AS mean
                        FROM    GENERATE_SERIES(0, group_size - 1) AS group_position
                        CROSS JOIN LATERAL
                                GENERATE_SUBSCRIPTS(_image, 2) iy
                        CROSS JOIN LATERAL
                                GENERATE_SUBSCRIPTS(_image, 3) ix
                        CROSS JOIN LATERAL
                                (
                                SELECT  _image[group_number + group_position + 1][iy][ix] AS value
                                )
                        ) q
                ) q1
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(l1) AS value
                FROM    GENERATE_SERIES(0, ARRAY_LENGTH(_image, 1) - 1) AS channel
                CROSS JOIN LATERAL
                        (
                        SELECT  ARRAY_AGG(l2) AS l1
                        FROM    GENERATE_SUBSCRIPTS(_image, 2) iy
                        CROSS JOIN LATERAL
                                (
                                SELECT  ARRAY_AGG(l3) AS l2
                                FROM    GENERATE_SUBSCRIPTS(_image, 3) ix
                                CROSS JOIN LATERAL
                                        (
                                        SELECT  (_image[channel + 1][iy][ix] - mean[channel / group_size + 1]) / var_denominator[channel / group_size + 1] *
                                                _weight[channel + 1] + _bias[channel + 1] AS l3
                                        )
                                ) q
                        ) q
                ) q2


        $$
        LANGUAGE SQL
        IMMUTABLE
        STRICT
        LEAKPROOF
        PARALLEL SAFE;

CREATE OR REPLACE FUNCTION RESNET
        (
        _input REAL[],
        _t_emb3 REAL[],
        _norm1_weight REAL[],
        _norm1_bias REAL[],
        _conv1_weight REAL[],
        _conv1_bias REAL[],
        _time_emb_proj_weight REAL[],
        _time_emb_proj_bias REAL[],
        _norm2_weight REAL[],
        _norm2_bias REAL[],
        _conv2_weight REAL[],
        _conv2_bias REAL[],
        _conv_shortcut_weight REAL[],
        _conv_shortcut_bias REAL[]
        )
RETURNS REAL[]
AS
$$

SELECT  hs9
FROM    (
        SELECT
        ) d
CROSS JOIN LATERAL
        GROUP_NORM(_input, _norm1_weight, _norm1_bias, 32) hs2
CROSS JOIN LATERAL
        SILU3(hs2) hs3
CROSS JOIN LATERAL
        CONV2D3X3(hs3, _conv1_weight, _conv1_bias) hs4
CROSS JOIN LATERAL
        LINEAR_1_2(_t_emb3, _time_emb_proj_weight, _time_emb_proj_bias) t_emb4
CROSS JOIN LATERAL
        PLUS_3_1(hs4, t_emb4) hs5
CROSS JOIN LATERAL
        GROUP_NORM(hs5, _norm2_weight, _norm2_bias, 32) hs6
CROSS JOIN LATERAL
        SILU3(hs6) AS hs7
CROSS JOIN LATERAL
        CONV2D3X3(hs7, _conv2_weight, _conv2_bias) hs8
CROSS JOIN LATERAL
        (SELECT COALESCE(CONV2D1X1(_input, _conv_shortcut_weight, _conv_shortcut_bias), _input) input2) input2
CROSS JOIN LATERAL
        PLUS_3_3(input2, hs8) hs9

$$
LANGUAGE SQL
IMMUTABLE
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION RESNET
        (
        _input REAL[],
        _t_emb3 REAL[],
        _prefix TEXT
        )
RETURNS REAL[]
AS
$$
SELECT  RESNET(
                _input,
                _t_emb3,
                norm1.weight,
                norm1.bias,
                conv1.weight,
                conv1.bias,
                time_emb_proj.weight,
                time_emb_proj.bias,
                norm2.weight,
                norm2.bias,
                conv2.weight,
                conv2.bias,
                conv_shortcut.weight,
                conv_shortcut.bias
        ) rn2
FROM    (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.norm1'
        ) norm1 (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.conv1'
        ) conv1 (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.time_emb_proj'
        ) time_emb_proj (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.norm2'
        ) norm2 (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.conv2'
        ) conv2 (weight, bias)
LEFT JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.conv_shortcut'
        ) conv_shortcut (weight, bias)
ON      TRUE
$$
LANGUAGE SQL
STRICT
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION SCALED_DOT_PRODUCTION_ATTENTION
        (
        _query REAL[],
        _key REAL[],
        _value REAL[],
        _head_dim INT
        )
RETURNS REAL[]
AS
$$

SELECT  scaled_reshaped
FROM    (
        SELECT  ARRAY_LENGTH(_query, 2) / _head_dim AS heads,
                1 / SQRT(_head_dim) AS scale_factor
        ) c
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS attn_weight
        FROM    GENERATE_SERIES(0, heads - 1) head
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v3) AS v2
                FROM    GENERATE_SUBSCRIPTS(_query, 1) qy
                CROSS JOIN LATERAL
                        (
                        SELECT  ARRAY_AGG(vexp) AS vexps
                        FROM    GENERATE_SUBSCRIPTS(_key, 1) ky
                        CROSS JOIN LATERAL
                                (
                                SELECT  EXP(SUM(_query[qy][head * _head_dim + x + 1] * _key[ky][head * _head_dim + x + 1]) * scale_factor) AS vexp
                                FROM    GENERATE_SERIES(0, _head_dim - 1) x
                                ) l4
                        ) l3
                CROSS JOIN LATERAL
                        (
                        SELECT  SUM(vexp) AS denominator
                        FROM    UNNEST(vexps) vexp
                        ) q
                CROSS JOIN LATERAL
                        (
                        SELECT  ARRAY_AGG(vexp / denominator) AS v3
                        FROM    UNNEST(vexps) vexp
                        ) q2
                ) l2
        ) hs1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS scaled_reshaped
        FROM    GENERATE_SUBSCRIPTS(attn_weight, 2) y
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v3) AS v2
                FROM    GENERATE_SERIES(0, heads - 1) head
                CROSS JOIN LATERAL
                        GENERATE_SERIES(0, _head_dim - 1 + head - head) hc
                CROSS JOIN LATERAL
                        (
                        SELECT  SUM(attn_weight[head + 1][y][ax] * _value[ax][head * _head_dim + hc + 1]) AS v3
                        FROM    GENERATE_SUBSCRIPTS(attn_weight, 3) ax
                        ) l3
                ) l2
        ) hs2
$$
LANGUAGE SQL
IMMUTABLE
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION ATTN
        (
        _input REAL[],
        _group_norm_weight REAL[],
        _group_norm_bias REAL[],
        _to_q_weight REAL[],
        _to_q_bias REAL[],
        _to_k_weight REAL[],
        _to_k_bias REAL[],
        _to_v_weight REAL[],
        _to_v_bias REAL[],
        _to_out_weight REAL[],
        _to_out_bias REAL[]
        )
RETURNS REAL[]
AS
$$

SELECT  attn
FROM    (
        SELECT  ARRAY_LENGTH(_input, 1) AS channel,
                ARRAY_LENGTH(_input, 2) AS height,
                ARRAY_LENGTH(_input, 3) AS width
        ) c
CROSS JOIN LATERAL
        GROUP_NORM(_input, _group_norm_weight, _group_norm_bias, 32) hs2
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS hs3
        FROM    GENERATE_SUBSCRIPTS(hs2, 2) l2
        CROSS JOIN LATERAL
                GENERATE_SUBSCRIPTS(hs2, 3 + (l2 - l2)) l3
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(hs2[l1][l2][l3]) v2
                FROM    GENERATE_SUBSCRIPTS(hs2, 1) l1
                )
        ) hs3
CROSS JOIN LATERAL
        LINEAR_2_2(hs3, _to_q_weight, _to_q_bias) query
CROSS JOIN LATERAL
        LINEAR_2_2(hs3, _to_k_weight, _to_k_bias) key
CROSS JOIN LATERAL
        LINEAR_2_2(hs3, _to_v_weight, _to_v_bias) value
CROSS JOIN LATERAL
        SCALED_DOT_PRODUCTION_ATTENTION(query, key, value, 8) scaled_reshaped
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS hs4
        FROM    GENERATE_SUBSCRIPTS(_to_out_weight, 1) oy
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v3) AS v2
                FROM    GENERATE_SERIES(0, height - 1) ch
                CROSS JOIN LATERAL
                        (
                        SELECT  ARRAY_AGG(v4) AS v3
                        FROM    GENERATE_SERIES(0, width - 1) cw
                        CROSS JOIN LATERAL
                                (
                                SELECT  ch * height + cw + 1 AS scy
                                ) q
                        CROSS JOIN LATERAL
                                (
                                SELECT  SUM(scaled_reshaped[scy][x] * _to_out_weight[oy][x]) + _to_out_bias[oy] AS v4
                                FROM    GENERATE_SUBSCRIPTS(scaled_reshaped, 2) x
                                ) l4
                        ) l3
                ) l2 
        ) hs4
CROSS JOIN LATERAL
        PLUS_3_3(hs4, _input) AS attn

$$
LANGUAGE SQL
IMMUTABLE
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION ATTN
        (
        _input REAL[],
        _prefix TEXT
        )
RETURNS REAL[]
AS
$$
SELECT  ATTN(
                _input,
                group_norm.weight,
                group_norm.bias,
                to_q.weight,
                to_q.bias,
                to_k.weight,
                to_k.bias,
                to_v.weight,
                to_v.bias,
                to_out.weight,
                to_out.bias
        ) rn2
FROM    (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.group_norm'
        ) group_norm (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.to_q'
        ) to_q (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.to_k'
        ) to_k (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.to_v'
        ) to_v (weight, bias)
CROSS JOIN
        (
        SELECT  weight::TEXT::REAL[], bias::TEXT::REAL[]
        FROM    parameters
        WHERE   key = _prefix || '.to_out.0'
        ) to_out (weight, bias)
$$
LANGUAGE SQL
STRICT
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION UPSAMPLE_NEAREST2D_3_SCALE_2(_image REAL[][][])
RETURNS REAL[][][]
AS
$$

SELECT  ARRAY_AGG(v1)
FROM    GENERATE_SUBSCRIPTS(_image, 1) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS v1
        FROM    GENERATE_SUBSCRIPTS(_image, 2) l2
        CROSS JOIN LATERAL
                GENERATE_SERIES(0, 1 + l2 - l2)
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(_image[l1][l2][l3]) AS v2
                FROM    GENERATE_SUBSCRIPTS(_image, 3) l3
                CROSS JOIN LATERAL
                        GENERATE_SERIES(0, 1 + l3 - l3)
                ) l3
        ) l2

$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION UNET(_image REAL[][][], _step INT)
RETURNS REAL[][][]
AS
$$

SELECT  out
FROM    EMB(_step) t_emb
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'time_embedding.linear_1') t_embp1 (weight, bias)
CROSS JOIN LATERAL LINEAR_1_2(t_emb, t_embp1.weight, t_embp1.bias) t_emb2
CROSS JOIN SILU(t_emb2) t_emb3
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'time_embedding.linear_2') t_embp2 (weight, bias)
CROSS JOIN LATERAL LINEAR_1_2(t_emb3, t_embp2.weight, t_embp2.bias) t_emb4
CROSS JOIN SILU(t_emb4) t_emb5
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'conv_in') conv_in (weight, bias)
CROSS JOIN LATERAL CONV2D3X3(_image, conv_in.weight, conv_in.bias) input1

-- Down 1
        
CROSS JOIN LATERAL RESNET(input1, t_emb5, 'down_blocks.0.resnets.0') db0rn0
CROSS JOIN LATERAL RESNET(db0rn0, t_emb5, 'down_blocks.0.resnets.1') db0rn1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'down_blocks.0.downsamplers.0.conv') db0ds (weight, bias)
CROSS JOIN LATERAL CONV2D3X3(db0rn1, db0ds.weight, db0ds.bias, 2) AS db0

-- Down 2

CROSS JOIN LATERAL RESNET(db0, t_emb5, 'down_blocks.1.resnets.0') db1rn0
CROSS JOIN LATERAL RESNET(db1rn0, t_emb5, 'down_blocks.1.resnets.1') db1rn1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'down_blocks.1.downsamplers.0.conv') db1ds (weight, bias)
CROSS JOIN LATERAL CONV2D3X3(db1rn1, db1ds.weight, db1ds.bias, 2) AS db1

-- Down 3 with Attention

CROSS JOIN LATERAL RESNET(db1, t_emb5, 'down_blocks.2.resnets.0') db2rn0
CROSS JOIN LATERAL ATTN(db2rn0, 'down_blocks.2.attentions.0') db2att0
CROSS JOIN LATERAL RESNET(db2att0, t_emb5, 'down_blocks.2.resnets.1') db2rn1
CROSS JOIN LATERAL ATTN(db2rn1, 'down_blocks.2.attentions.1') db2att1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'down_blocks.2.downsamplers.0.conv') db2ds (weight, bias)
CROSS JOIN LATERAL CONV2D3X3(db2att1, db2ds.weight, db2ds.bias, 2) AS db2

-- Down 4 with Attention

CROSS JOIN LATERAL RESNET(db2, t_emb5, 'down_blocks.3.resnets.0') db3rn0
CROSS JOIN LATERAL ATTN(db3rn0, 'down_blocks.3.attentions.0') db3att0
CROSS JOIN LATERAL RESNET(db3att0, t_emb5, 'down_blocks.3.resnets.1') db3rn1
CROSS JOIN LATERAL ATTN(db3rn1, 'down_blocks.3.attentions.1') db3

-- Mid

CROSS JOIN LATERAL RESNET(db3, t_emb5, 'mid_block.resnets.0') mbrn0
CROSS JOIN LATERAL ATTN(mbrn0, 'mid_block.attentions.0') mbatt0
CROSS JOIN LATERAL RESNET(mbatt0, t_emb5, 'mid_block.resnets.1') mb

-- Up 1 with Attention

CROSS JOIN LATERAL RESNET(mb || db3, t_emb5, 'up_blocks.0.resnets.0') ub0rn0
CROSS JOIN LATERAL ATTN(ub0rn0, 'up_blocks.0.attentions.0') ub0att0
CROSS JOIN LATERAL RESNET(ub0att0 || db3att0, t_emb5, 'up_blocks.0.resnets.1') ub0rn1
CROSS JOIN LATERAL ATTN(ub0rn1, 'up_blocks.0.attentions.1') ub0att1
CROSS JOIN LATERAL RESNET(ub0att1 || db2, t_emb5, 'up_blocks.0.resnets.2') ub0rn2
CROSS JOIN LATERAL ATTN(ub0rn2, 'up_blocks.0.attentions.2') ub0att2
CROSS JOIN UPSAMPLE_NEAREST2D_3_SCALE_2(ub0att2) ub0us1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'up_blocks.0.upsamplers.0.conv') ub0us (weight, bias)
CROSS JOIN CONV2D3X3(ub0us1, ub0us.weight, ub0us.bias) ub0

-- Up 2 with Attention

CROSS JOIN LATERAL RESNET(ub0 || db2att1, t_emb5, 'up_blocks.1.resnets.0') ub1rn0
CROSS JOIN LATERAL ATTN(ub1rn0, 'up_blocks.1.attentions.0') ub1att0
CROSS JOIN LATERAL RESNET(ub1att0 || db2att0, t_emb5, 'up_blocks.1.resnets.1') ub1rn1
CROSS JOIN LATERAL ATTN(ub1rn1, 'up_blocks.1.attentions.1') ub1att1
CROSS JOIN LATERAL RESNET(ub1att1 || db1, t_emb5, 'up_blocks.1.resnets.2') ub1rn2
CROSS JOIN LATERAL ATTN(ub1rn2, 'up_blocks.1.attentions.2') ub1att2
CROSS JOIN UPSAMPLE_NEAREST2D_3_SCALE_2(ub1att2) ub1us1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'up_blocks.1.upsamplers.0.conv') ub1us (weight, bias)
CROSS JOIN CONV2D3X3(ub1us1, ub1us.weight, ub1us.bias) ub1

-- Up 3

CROSS JOIN LATERAL RESNET(ub1 || db1rn1, t_emb5, 'up_blocks.2.resnets.0') ub2rn0
CROSS JOIN LATERAL RESNET(ub2rn0 || db1rn0, t_emb5, 'up_blocks.2.resnets.1') ub2rn1
CROSS JOIN LATERAL RESNET(ub2rn1 || db0, t_emb5, 'up_blocks.2.resnets.2') ub2rn2
CROSS JOIN UPSAMPLE_NEAREST2D_3_SCALE_2(ub2rn2) ub2us1
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'up_blocks.2.upsamplers.0.conv') ub2us (weight, bias)
CROSS JOIN CONV2D3X3(ub2us1, ub2us.weight, ub2us.bias) ub2

-- Up 4

CROSS JOIN LATERAL RESNET(ub2 || db0rn1, t_emb5, 'up_blocks.3.resnets.0') ub3rn0
CROSS JOIN LATERAL RESNET(ub3rn0 || db0rn0, t_emb5, 'up_blocks.3.resnets.1') ub3rn1
CROSS JOIN LATERAL RESNET(ub3rn1 || input1, t_emb5, 'up_blocks.3.resnets.2') ub3

-- Decode

CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'conv_norm_out') conv_norm_out (weight, bias)
CROSS JOIN LATERAL GROUP_NORM(ub3, conv_norm_out.weight, conv_norm_out.bias, 32) conv_norm_out1
CROSS JOIN LATERAL SILU3(conv_norm_out1) conv_norm_out2
CROSS JOIN (SELECT weight::TEXT::REAL[], bias::TEXT::REAL[] FROM parameters WHERE key = 'conv_out') conv_out (weight, bias)
CROSS JOIN LATERAL CONV2D3X3(conv_norm_out2, conv_out.weight, conv_out.bias) out

$$
LANGUAGE SQL
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION PREDICT_PREVIOUS(_image REAL[][][], _noise REAL[][][], _t INT, _t2 INT)
RETURNS REAL[][][]
AS
$$
WITH    alphas_p AS
        (
        SELECT  EXP(SUM(LN(1 - ((0.02 - 0.0001) * step / 999 + 0.0001))) OVER (ORDER BY step)) AS alphas_p,
                step
        FROM    GENERATE_SERIES(0, 999) step
        )
SELECT  v1
FROM    (
        SELECT  alphas_p AS alphas_p_t
        FROM    alphas_p
        WHERE   step = _t
        ) q_t
CROSS JOIN LATERAL
        (
        SELECT  alphas_p AS alphas_p_t2
        FROM    alphas_p
        WHERE   step = _t2
        ) q_t2
CROSS JOIN LATERAL
        (
        SELECT  SQRT(alphas_p_t2) * (1 - alphas_p_t / alphas_p_t2) / (1 - alphas_p_t) AS x0_c,
                SQRT(alphas_p_t / alphas_p_t2) * (1 - alphas_p_t2) / (1 - alphas_p_t) AS xt2_c
        )
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v2) AS v1
        FROM    GENERATE_SUBSCRIPTS(_image, 1) l1
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v3) AS v2
                FROM    GENERATE_SUBSCRIPTS(_image, 2) l2
                CROSS JOIN LATERAL
                        (
                        SELECT  ARRAY_AGG(image_t2) AS v3
                        FROM    GENERATE_SUBSCRIPTS(_image, 3) l3
                        CROSS JOIN LATERAL
                                (
                                        SELECT  _image[l1][l2][l3] AS image,
                                                _noise[l1][l2][l3] AS noise
                                ) q
                        CROSS JOIN LATERAL
                                (
                                SELECT  GREATEST(LEAST((image - noise * SQRT(1 - alphas_p_t)) / SQRT(alphas_p_t), 1), -1) AS x0
                                ) x0
                        CROSS JOIN LATERAL
                                (
                                SELECT  x0_c * x0 + xt2_c * image AS image_t2
                                ) image_t2
                        ) l3
                ) l2
        ) l1
$$
LANGUAGE SQL
IMMUTABLE
STRICT
LEAKPROOF
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION GENERATE_NOISE()
RETURNS REAL[][][]
AS
$$
WITH    initial_image (input) AS MATERIALIZED
        (
        SELECT  ARRAY_AGG(v)
        FROM    GENERATE_SERIES(1, 3) l1
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG(v) v
                FROM    GENERATE_SERIES(1, 64) l2
                CROSS JOIN LATERAL
                        (
                        -- Introduce dependency, otherwise GENERATE_SERIES(1, 64) will be cached and generate duplicates
                        -- The distribution is uniform, but has the same mean and variance as the standard normal distribution
                        SELECT  ARRAY_AGG((RANDOM()::REAL * 2 - 1) * 1.73 + l1 + l2 + l3 - l1 - l2 - l3) v
                        FROM    GENERATE_SERIES(1, 64) l3
                        ) l3
                ) l3
        )
SELECT  input
FROM    initial_image
$$
LANGUAGE SQL;

CREATE OR REPLACE FUNCTION GENERATE_CREATIVE_NOISE(_t INT, _t2 INT)
RETURNS REAL[][][]
AS
$$
WITH    alphas_p AS
        (
        SELECT  EXP(SUM(LN(1 - ((0.02 - 0.0001) * step / 999 + 0.0001))) OVER (ORDER BY step)) AS alphas_p,
                step
        FROM    GENERATE_SERIES(0, 999) step
        )
SELECT  ARRAY_AGG(v)
FROM    (
        SELECT  alphas_p AS alphas_p_t
        FROM    alphas_p
        WHERE   step = _t
        ) q_t
CROSS JOIN LATERAL
        (
        SELECT  alphas_p AS alphas_p_t2
        FROM    alphas_p
        WHERE   step = _t2
        ) q_t2
CROSS JOIN LATERAL
        (
        SELECT  SQRT((1 - alphas_p_t2)  / (1 - alphas_p_t) * (1 - alphas_p_t / alphas_p_t2)) AS creative_noise_c
        ) q1
CROSS JOIN LATERAL
        GENERATE_SERIES(1, 3) l1
CROSS JOIN LATERAL
        (
        SELECT  ARRAY_AGG(v) v
        FROM    GENERATE_SERIES(1, 64) l2
        CROSS JOIN LATERAL
                (
                SELECT  ARRAY_AGG((RANDOM()::REAL * 2 - 1) * 1.73 * creative_noise_c + l1 + l2 + l3 - l1 - l2 - l3) v
                FROM    GENERATE_SERIES(1, 64) l3
                ) l3
        ) l3
$$
LANGUAGE SQL;
