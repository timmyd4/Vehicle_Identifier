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
        pipeline = new CarPipeline(models, MODELS, 0.7f);
    }

    @AfterAll
    static void closeModels() throws Exception {
        if (models != null) {
            models.close();
        }
    }

    /**
     * GOLDEN VALUES: expected outputs saved from a known-good run. They belong to ONE model,
     * so they must be re-made after every retrain. When this test fails right after a new
     * export, that's the test correctly noticing the model changed, not necessarily a bug.
     *
     * <p>Run 2 (generation labels, 2026-10-26): Java says Toyota Corolla (2009-2010) 91.4%,
     * and predict.py says 91.2% with the same top 5 in the same order. The car is a 2009,
     * so the full layered answer (E140, pre-facelift) is right.
     *
     * <p>Why Java and predict.py differ slightly: Java's JPEG decoder (ImageIO) and Python's
     * (PIL / libjpeg-turbo) turn the same file into slightly different pixels, mostly in
     * color detail (measured 2026-10-25: 70% of values differ, by ~2 levels on average). On
     * identical pixels the two pipelines matched exactly (checked with run 1). It only
     * matters when guesses are nearly tied, as run 1's were on this photo (5.9% vs 5.7%).
     */
    @Test
    void corollaMatchesPython() throws Exception {
        List<CarResult> results = pipeline.identify(readPhoto("corolla_2009.jpg"), true);
        print("corolla_2009.jpg", results);

        assertFalse(results.isEmpty());
        CarResult biggest = results.get(0);   // Python only looks at the biggest vehicle
        assertTrue(biggest.detected(), "detector should find the car");
        assertEquals("car", biggest.vehicleType());
        assertEquals(0.94, biggest.detectorScore(), 0.005);

        List<Prediction> top = biggest.predictions();
        assertEquals("Toyota Corolla (2009-2010)", top.get(0).name());
        assertEquals(0.914, top.get(0).probability(), 0.005);
        assertEquals("Scion TC (2005-2007)", top.get(1).name());

        LayeredAnswer answer = biggest.answer();
        assertEquals(LayeredAnswer.Level.LOOK, answer.level());
        assertEquals("Toyota Corolla, E140, pre-facelift (2009-2010)", answer.text());
    }

    @Test
    void highlanderFindsAVehicle() throws Exception {
        // No exact check. A 2017 Highlander is the XU50 generation, which the model has no
        // class for (VMMRdb ends in 2016). Run 2 answers the nearest thing it knows, the
        // XU40 facelift, at 75%: right model, wrong generation, and sure of it. That's
        // open-set recognition failing. Newer training photos (Commons) are the real fix;
        // keep this photo as a check for run 3.
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
            System.out.printf("  answer: %s  [%s, %.1f%%]%n", r.answer().text(), r.answer().level(),
                    r.answer().confidence() * 100);
            int rank = 1;
            for (Prediction p : r.predictions()) {
                System.out.printf("    %d. %-42s %5.1f%%%n", rank++, p.name(), p.probability() * 100);
            }
        }
    }
}
