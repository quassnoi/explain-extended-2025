SELECT SETSEED(0.20241231);

WITH    initial_image (input) AS MATERIALIZED
        (
        SELECT  GENERATE_NOISE()
        )
SELECT  unet[1:2][1:2][1:2], ARRAY_DIMS(unet)
FROM    UNET((SELECT input FROM initial_image), 999) unet
