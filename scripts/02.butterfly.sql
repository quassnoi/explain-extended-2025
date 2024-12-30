WITH    initial_image (input) AS MATERIALIZED
        (
        SELECT  GENERATE_NOISE()
        )
SELECT  v2
FROM    (SELECT)
CROSS JOIN LATERAL UNET((SELECT input FROM initial_image), 500) unet1
CROSS JOIN LATERAL PREDICT_PREVIOUS((SELECT input FROM initial_image), unet1, 500, 250) step1
CROSS JOIN LATERAL GENERATE_CREATIVE_NOISE(500, 250) noise1
CROSS JOIN PLUS_3_3(step1, noise1) image2
CROSS JOIN LATERAL UNET(image2, 250) unet2
CROSS JOIN LATERAL PREDICT_PREVIOUS(image2, unet2, 250, 125) step2
CROSS JOIN LATERAL GENERATE_CREATIVE_NOISE(250, 125) noise2
CROSS JOIN PLUS_3_3(step2, noise2) image3
CROSS JOIN LATERAL UNET(image3, 125) unet3
CROSS JOIN LATERAL PREDICT_PREVIOUS(image3, unet3, 125, 60) step3
CROSS JOIN LATERAL GENERATE_CREATIVE_NOISE(125, 60) noise3
CROSS JOIN PLUS_3_3(step3, noise3) image4
CROSS JOIN LATERAL UNET(image4, 60) unet4
CROSS JOIN LATERAL PREDICT_PREVIOUS(image4, unet4, 60, 0) butterfly
CROSS JOIN LATERAL
        (
        SELECT  STRING_TO_ARRAY(' `.-'':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tluneoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#Bg0MNWQ%&@', NULL) AS codes
        )
CROSS JOIN LATERAL
        (
        SELECT  v2
        FROM    GENERATE_SUBSCRIPTS(butterfly, 2) y
        CROSS JOIN LATERAL
                (
                SELECT  STRING_AGG(code, '') v2
                FROM    GENERATE_SUBSCRIPTS(butterfly, 3) x
                CROSS JOIN LATERAL
                        (
                        SELECT  FLOOR(LEAST(GREATEST(AVG(-v) * 2 - 1 + 1, 0), 1) * (ARRAY_LENGTH(codes, 1) - 1))::INT + 1 AS brightness
                        FROM    UNNEST(butterfly[1:3][y:y][x:x]) v
                        )
                CROSS JOIN LATERAL
                        (
                        SELECT  codes[brightness] AS code
                        )
                )
        )