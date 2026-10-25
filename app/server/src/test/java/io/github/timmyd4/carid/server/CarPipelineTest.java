package io.github.timmyd4.carid.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.timmyd4.carid.server.CarResult.Prediction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Runs the real pipeline (real models, real photos) and compares with training/predict.py.
 *
 * <p>An "integration test": it exercises several pieces together (ONNX Runtime, both models,
 * preprocessing, labels) instead of one class alone. It needs the model files and your test
 * photos, which aren't in git, so it SKIPS itself when they're missing (e.g. on a fresh
 * clone) instead of failing.
 *
 * <p>Maven runs tests with the module folder (app/server) as the working directory, so the
 * project root is two levels up.
 */
class CarPipelineTest {

    private static final Path ROOT = Path.of("../..");
    private static final Path MODELS = ROOT.resolve("models");
    private static final Path PHOTOS = ROOT.resolve("data/my_photos");

    // static + @BeforeAll: load the models ONCE for all tests in this class (it takes ~1 s).
    private static OnnxModels models;
    private static CarPipeline pipeline;

    @BeforeAll
    static void loadModels() throws Exception {
        // assumeTrue: if false, the tests are reported as "skipped", not failed.
        assumeTrue(Files.exists(MODELS.resolve("car_model.onnx")), "models/ not found, skipping");
        // No Spring here: we build the two objects by hand, passing what Spring would inject.
        models = new OnnxModels(MODELS);
        pipeline = new CarPipeline(models, MODELS);
    }

    @AfterAll
    static void closeModels() throws Exception {
        if (models != null) {
            models.close();
        }
    }

    /**
     * Expected values come from Python's ONNX pipeline run on the SAME pixels Java decodes
     * (verified 2026-10-25): car 94%, then Mazda 3 Hatchback 5.9%, Toyota Corolla
     * (2010-2012) 4.0%, Toyota Corolla (2007-2009) 2.5%. Java matches exactly.
     *
     * <p>Why not compare with predict.py's output directly: Java's JPEG decoder (ImageIO)
     * and Python's (PIL / libjpeg-turbo) turn the same file into slightly different pixels,
     * mostly in color detail (measured: 70% of values differ, by ~2 levels on average). On
     * this photo the top two guesses are nearly tied, so predict.py ranks Corolla S first
     * (5.9% vs 5.7%). That's a decoder difference, not a pipeline bug; the pipeline itself
     * gives identical results on identical pixels.
     */
    @Test
    void corollaMatchesPythonOnSamePixels() throws Exception {
        List<CarResult> results = pipeline.identify(readPhoto("corolla_2009.jpg"), true);
        print("corolla_2009.jpg", results);

        assertFalse(results.isEmpty());
        CarResult biggest = results.get(0);   // Python only looks at the biggest vehicle
        assertTrue(biggest.detected(), "detector should find the car");
        assertEquals("car", biggest.vehicleType());
        assertEquals(0.94, biggest.detectorScore(), 0.005);

        List<Prediction> top = biggest.predictions();
        assertEquals("Mazda 3 Hatchback (2004-2006)", top.get(0).name());
        assertEquals(0.059, top.get(0).probability(), 0.001);
        assertEquals("Toyota Corolla (2010-2012)", top.get(1).name());
        assertEquals(0.040, top.get(1).probability(), 0.001);
        assertEquals("Toyota Corolla (2007-2009)", top.get(2).name());
    }

    @Test
    void highlanderFindsAVehicle() throws Exception {
        // No exact check: the 2017 Highlander has no class, so its guesses are all
        // low and close together. This one just shows the output and timings.
        List<CarResult> results = pipeline.identify(readPhoto("highlander_2017.jpg"), true);
        print("highlander_2017.jpg", results);
        assertTrue(results.get(0).detected());
    }

    private static RgbImage readPhoto(String name) throws IOException {
        Path file = PHOTOS.resolve(name);
        assumeTrue(Files.exists(file), file + " not found, skipping");
        return RgbImage.from(ImageIO.read(file.toFile()));
    }

    /** Show results in the test output, in the same style as predict.py. */
    private static void print(String photo, List<CarResult> results) {
        System.out.println("\n" + photo + ": " + results.size() + " vehicle(s)");
        for (CarResult r : results) {
            System.out.printf("  %s %.0f%% at %s%n", r.vehicleType(), r.detectorScore() * 100, r.box());
            int rank = 1;
            for (Prediction p : r.predictions()) {
                System.out.printf("    %d. %-42s %5.1f%%%n", rank++, p.name(), p.probability() * 100);
            }
        }
    }
}
