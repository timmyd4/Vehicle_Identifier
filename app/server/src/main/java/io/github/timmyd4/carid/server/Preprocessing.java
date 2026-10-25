package io.github.timmyd4.carid.server;

/**
 * Turns RGB images into the float arrays ("tensors") the two ONNX models expect.
 *
 * <p>Big picture: a model only gives correct answers if its input is prepared exactly the
 * way it was during training. Training happened in Python, so every step here mirrors
 * Python (see models/preprocessing.json, written by training/export_onnx.py):
 * <ul>
 *   <li>Classifier: resize to 224x224 with PIL's bilinear filter, scale to 0-1, then
 *       subtract the ImageNet mean and divide by the std, per channel.</li>
 *   <li>Detector: the whole image, just scaled to 0-1 (it resizes internally).</li>
 * </ul>
 * Both use "CHW" layout: all red values first, then all green, then all blue. Images are
 * stored pixel-by-pixel (RGBRGB...), so the conversion reorders them.
 */
public final class Preprocessing {

    // PIL stores filter weights as fixed-point integers with this many fractional bits
    // (32-bit int, minus 8 bits for the pixel value, minus 2 bits of headroom).
    private static final int PRECISION_BITS = 32 - 8 - 2;

    private Preprocessing() {
    }

    // ---------------------------------------------------------------- tensors

    /**
     * Classifier input: 3 x height x width floats, normalized as
     * {@code (pixel / 255 - mean[c]) / std[c]}. The image must already be resized.
     *
     * <p>The math is done in float (32-bit) in the same order as PyTorch's ToTensor then
     * Normalize, so the results match Python bit for bit, not just approximately.
     */
    public static float[] classifierTensor(RgbImage image, float[] mean, float[] std) {
        int plane = image.width() * image.height();
        float[] out = new float[3 * plane];
        byte[] px = image.pixels();
        for (int i = 0; i < plane; i++) {
            for (int c = 0; c < 3; c++) {
                float value = (px[i * 3 + c] & 0xFF) / 255f;
                out[c * plane + i] = (value - mean[c]) / std[c];
            }
        }
        return out;
    }

    /** Detector input: 3 x height x width floats, each pixel / 255 (0.0 to 1.0). */
    public static float[] detectorTensor(RgbImage image) {
        int plane = image.width() * image.height();
        float[] out = new float[3 * plane];
        byte[] px = image.pixels();
        for (int i = 0; i < plane; i++) {
            for (int c = 0; c < 3; c++) {
                out[c * plane + i] = (px[i * 3 + c] & 0xFF) / 255f;
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- PIL bilinear resize

    /**
     * Resize exactly like PIL's {@code image.resize((width, height), Image.BILINEAR)}.
     *
     * <p>How PIL resizes: it does two 1-D passes, first stretching every row to the new
     * width, then every column to the new height. Each output pixel is a weighted average
     * of nearby input pixels; the weights come from a "triangle" (tent) shape, highest for
     * the closest input pixel and falling to zero further away.
     *
     * <p>When shrinking, PIL widens the triangle by the shrink factor, so each output pixel
     * averages ALL the input pixels it covers. That's "antialiasing": without it, shrinking
     * a 4000-pixel-wide photo to 224 would just sample scattered pixels and look jagged.
     * Java's Graphics2D scaling doesn't do this, which is why we can't use it.
     */
    public static RgbImage resizeBilinear(RgbImage in, int outWidth, int outHeight) {
        // Like PIL, skip a pass when that dimension doesn't change.
        RgbImage wide = in.width() == outWidth ? in : resizeRows(in, outWidth);
        return wide.height() == outHeight ? wide : resizeColumns(wide, outHeight);
    }

    /**
     * For each output position along one dimension: which input pixel to start at, and the
     * integer weight for each input pixel it averages. Computed once per resize and reused
     * for every row (or column), since every row shrinks the same way.
     */
    private record Weights(int[] start, int[][] fixed) {
    }

    /** Port of PIL's precompute_coeffs() + normalize_coeffs_8bpc() (libImaging/Resample.c). */
    private static Weights computeWeights(int inSize, int outSize) {
        double scale = (double) inSize / outSize;
        // Growing: the triangle is 1 input pixel wide on each side. Shrinking: wider by the factor.
        double filterScale = Math.max(scale, 1.0);
        double support = filterScale;   // bilinear's triangle has radius 1, times filterScale
        // PIL multiplies by 1/filterScale instead of dividing by filterScale. In floating point
        // those can differ in the last bit, so we copy PIL's form exactly.
        double inverseScale = 1.0 / filterScale;
        int[] start = new int[outSize];
        int[][] fixed = new int[outSize][];

        for (int out = 0; out < outSize; out++) {
            // Where this output pixel's center falls in input coordinates.
            double center = (out + 0.5) * scale;
            // Range of input pixels under the triangle, clamped to the image.
            // (int) truncates toward zero, the same as C's cast in PIL.
            int min = Math.max(0, (int) (center - support + 0.5));
            int max = Math.min(inSize, (int) (center + support + 0.5));
            int count = max - min;

            double[] w = new double[count];
            double total = 0;
            for (int i = 0; i < count; i++) {
                // Distance from the output center to this input pixel's center, in triangle units.
                w[i] = triangle((min + i - center + 0.5) * inverseScale);
                total += w[i];
            }

            // Make the weights add up to 1, then store them as fixed-point integers:
            // weight * 2^22, rounded. PIL uses integers so its results are exact and fast;
            // matching PIL means using the same integers, not doubles.
            int[] k = new int[count];
            for (int i = 0; i < count; i++) {
                double normalized = total != 0 ? w[i] / total : w[i];
                double scaled = normalized * (1 << PRECISION_BITS);
                k[i] = (int) (scaled < 0 ? scaled - 0.5 : scaled + 0.5);
            }
            start[out] = min;
            fixed[out] = k;
        }
        return new Weights(start, fixed);
    }

    /** PIL's bilinear filter: 1 at distance 0, falling linearly to 0 at distance 1. */
    private static double triangle(double x) {
        x = Math.abs(x);
        return x < 1.0 ? 1.0 - x : 0.0;
    }

    /** Horizontal pass: every row becomes outWidth pixels wide. */
    private static RgbImage resizeRows(RgbImage in, int outWidth) {
        Weights weights = computeWeights(in.width(), outWidth);
        int h = in.height();
        byte[] src = in.pixels();
        byte[] out = new byte[outWidth * h * 3];
        for (int y = 0; y < h; y++) {
            int rowStart = y * in.width();
            for (int x = 0; x < outWidth; x++) {
                int[] k = weights.fixed()[x];
                int first = rowStart + weights.start()[x];
                for (int c = 0; c < 3; c++) {
                    // Start at "0.5" in fixed point so the final shift rounds instead of truncating.
                    int sum = 1 << (PRECISION_BITS - 1);
                    for (int i = 0; i < k.length; i++) {
                        sum += (src[(first + i) * 3 + c] & 0xFF) * k[i];
                    }
                    out[(y * outWidth + x) * 3 + c] = clip8(sum);
                }
            }
        }
        return new RgbImage(outWidth, h, out);
    }

    /** Vertical pass: every column becomes outHeight pixels tall. Same math, other direction. */
    private static RgbImage resizeColumns(RgbImage in, int outHeight) {
        Weights weights = computeWeights(in.height(), outHeight);
        int w = in.width();
        byte[] src = in.pixels();
        byte[] out = new byte[w * outHeight * 3];
        for (int y = 0; y < outHeight; y++) {
            int[] k = weights.fixed()[y];
            int firstRow = weights.start()[y];
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    int sum = 1 << (PRECISION_BITS - 1);
                    for (int i = 0; i < k.length; i++) {
                        sum += (src[((firstRow + i) * w + x) * 3 + c] & 0xFF) * k[i];
                    }
                    out[(y * w + x) * 3 + c] = clip8(sum);
                }
            }
        }
        return new RgbImage(w, outHeight, out);
    }

    /** Fixed-point sum back to a 0-255 byte: drop the fractional bits, clamp to range. */
    private static byte clip8(int sum) {
        int v = sum >> PRECISION_BITS;
        return (byte) (v < 0 ? 0 : Math.min(v, 255));
    }
}
