package io.github.timmyd4.carid.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import javax.imageio.ImageIO;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Proves Java preprocessing produces exactly what Python produced during training.
 *
 * <p>The expected outputs in src/test/resources/parity/ were written by
 * training/make_parity_fixture.py using the real Python evaluation recipe. If someone
 * changes Preprocessing.java (or Python's recipe changes and the fixture is regenerated),
 * this test catches any difference before it silently lowers accuracy.
 *
 * <p>A plain unit test: no Spring, no models, no server. It runs in milliseconds.
 */
class PreprocessingParityTest {

    // Must match models/preprocessing.json (and therefore the fixture).
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final int SIZE = 224;

    // @ParameterizedTest + @ValueSource: run this test once per fixture case.
    // Each case shows up separately in the results, like two test methods.
    @ParameterizedTest
    @ValueSource(strings = {"shrink", "grow"})
    void resizeMatchesPilExactly(String name) throws IOException {
        RgbImage input = readPng(name + "_input.png");
        RgbImage expected = readPng(name + "_resized.png");

        RgbImage actual = Preprocessing.resizeBilinear(input, SIZE, SIZE);

        assertEquals(SIZE, actual.width());
        assertEquals(SIZE, actual.height());
        // Every byte must be identical: not "close", identical.
        assertArrayEquals(expected.pixels(), actual.pixels(),
                () -> name + ": resized pixels differ from PIL (" + countDiffs(expected, actual) + " bytes)");
    }

    @ParameterizedTest
    @ValueSource(strings = {"shrink", "grow"})
    void classifierTensorMatchesTorchvision(String name) throws IOException {
        RgbImage input = readPng(name + "_input.png");
        float[] expected = readFloats(name + "_tensor.f32");

        float[] actual = Preprocessing.classifierTensor(
                Preprocessing.resizeBilinear(input, SIZE, SIZE), MEAN, STD);

        assertEquals(3 * SIZE * SIZE, actual.length);
        // Tolerance 0: we expect bit-identical floats, since both sides do the same
        // float32 operations in the same order.
        assertArrayEquals(expected, actual, 0f, name + ": classifier tensor differs from torchvision");
    }

    // ------------------------------------------------------------ helpers

    /** Load a fixture file from the test classpath (src/test/resources/parity/). */
    private static InputStream open(String file) {
        InputStream in = PreprocessingParityTest.class.getResourceAsStream("/parity/" + file);
        if (in == null) {
            throw new IllegalStateException("Missing fixture /parity/" + file
                    + " (run python training/make_parity_fixture.py)");
        }
        return in;
    }

    private static RgbImage readPng(String file) throws IOException {
        try (InputStream in = open(file)) {
            return RgbImage.from(ImageIO.read(in));
        }
    }

    /** Raw little-endian float32 values, as written by numpy's astype("<f4").tobytes(). */
    private static float[] readFloats(String file) throws IOException {
        try (InputStream in = open(file)) {
            ByteBuffer bytes = ByteBuffer.wrap(in.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN);
            float[] out = new float[bytes.remaining() / 4];
            bytes.asFloatBuffer().get(out);
            return out;
        }
    }

    private static int countDiffs(RgbImage a, RgbImage b) {
        int diffs = 0;
        for (int i = 0; i < a.pixels().length; i++) {
            if (a.pixels()[i] != b.pixels()[i]) {
                diffs++;
            }
        }
        return diffs;
    }
}
