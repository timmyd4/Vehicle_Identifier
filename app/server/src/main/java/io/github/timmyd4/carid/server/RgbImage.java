package io.github.timmyd4.carid.server;

import java.awt.image.BufferedImage;

/**
 * A photo as plain RGB bytes: the one image type the rest of the pipeline works with.
 *
 * <p>Why not just use Java's BufferedImage everywhere: BufferedImage has many internal
 * layouts (BGR, ARGB, gray, indexed...) and its scaling methods use different math than
 * Python's PIL. Converting once into a simple, predictable array lets us control every
 * step ourselves and match Python exactly.
 *
 * <p>Layout: R, G, B bytes per pixel, row after row (same as PIL's "RGB" mode).
 * The pixel at (x, y) starts at index (y * width + x) * 3.
 *
 * @param pixels values 0-255 stored in signed Java bytes; read them with {@code & 0xFF}
 */
public record RgbImage(int width, int height, byte[] pixels) {

    // A "compact constructor": runs before the record's fields are set. Used to validate.
    public RgbImage {
        if (width <= 0 || height <= 0 || pixels.length != width * height * 3) {
            throw new IllegalArgumentException("Bad image size " + width + "x" + height
                    + " for " + pixels.length + " bytes");
        }
    }

    /** Copy any BufferedImage (as decoded by ImageIO) into RGB bytes. */
    public static RgbImage from(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        byte[] out = new byte[w * h * 3];
        int[] row = new int[w];
        int i = 0;
        for (int y = 0; y < h; y++) {
            // getRGB converts whatever internal layout the image has into packed 0xAARRGGBB ints.
            image.getRGB(0, y, w, 1, row, 0, w);
            for (int x = 0; x < w; x++) {
                int argb = row[x];
                out[i++] = (byte) (argb >> 16);  // red   (casting to byte keeps the low 8 bits)
                out[i++] = (byte) (argb >> 8);   // green
                out[i++] = (byte) argb;          // blue
            }
        }
        return new RgbImage(w, h, out);
    }

    /** One channel value 0-255 (channel 0 = R, 1 = G, 2 = B). */
    public int get(int x, int y, int channel) {
        // & 0xFF turns Java's signed byte (-128..127) back into 0..255.
        return pixels[(y * width + x) * 3 + channel] & 0xFF;
    }

    /**
     * Cut out a rectangle, like PIL's image.crop((left, top, right, bottom)).
     * right and bottom are exclusive, so the result is (right - left) x (bottom - top).
     */
    public RgbImage crop(int left, int top, int right, int bottom) {
        if (left < 0 || top < 0 || right > width || bottom > height || left >= right || top >= bottom) {
            throw new IllegalArgumentException("Crop box outside the image: "
                    + left + "," + top + "," + right + "," + bottom + " in " + width + "x" + height);
        }
        int w = right - left;
        int h = bottom - top;
        byte[] out = new byte[w * h * 3];
        for (int y = 0; y < h; y++) {
            // Each row of the crop is one continuous run of bytes in the source.
            System.arraycopy(pixels, ((top + y) * width + left) * 3, out, y * w * 3, w * 3);
        }
        return new RgbImage(w, h, out);
    }
}
