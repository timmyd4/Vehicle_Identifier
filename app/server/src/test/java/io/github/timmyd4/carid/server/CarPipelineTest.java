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
     * <p>Run 4 (facelift gap filled, 2026-10-28): Java says Toyota Corolla (2009-2010) 92.5%,
     * predict.py 90.5%. (Run 3: Java 96.0%. Run 2: 91.4%.) The car is a 2009, so the full
     * layered answer (E140, pre-facelift) is right. Second place isn't checked any more: every
     * runner-up is ~0.0%, so their order is decoding noise (Java: Firebird, Python: Corolla).
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
        assertEquals(0.925, top.get(0).probability(), 0.005);

        LayeredAnswer answer = biggest.answer();
        assertEquals(LayeredAnswer.Level.LOOK, answer.level());
        assertEquals("Toyota Corolla, E140, pre-facelift (2009-2010)", answer.text());
    }

    @Test
    void highlanderFaceliftIsRight() throws Exception {
        // A 2017 Highlander is the XU50 FACELIFT (2017-2019). The history of this photo:
        //   run 2: XU40 facelift 75%       wrong generation (no XU50 class at all)
        //   run 3: XU50 pre-facelift 75.9% right generation, but no facelift class yet
        //   run 4: XU50 facelift (predict.py 95.8%), after 48 hand-labeled facelift photos
        //          on commons_looks.py brought the look to 58 photos (50 makes a class).
        // Only the answer is checked, not the exact probability: it's far from any tie.
        List<CarResult> results = pipeline.identify(readPhoto("highlander_2017.jpg"), true);
        print("highlander_2017.jpg", results);

        CarResult biggest = results.get(0);
        assertTrue(biggest.detected());
        assertEquals("Toyota Highlander (2017-2019)", biggest.predictions().get(0).name());
        assertEquals(LayeredAnswer.Level.LOOK, biggest.answer().level());
        assertEquals("Toyota Highlander, XU50, facelift (2017-2019)", biggest.answer().text());
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
