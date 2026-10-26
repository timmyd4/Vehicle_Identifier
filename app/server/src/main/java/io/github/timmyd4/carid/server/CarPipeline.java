package io.github.timmyd4.carid.server;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.timmyd4.carid.server.CarResult.Box;
import io.github.timmyd4.carid.server.CarResult.Prediction;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import tools.jackson.databind.json.JsonMapper;

/**
 * The full recognition pipeline: photo in, every vehicle found plus what car it is out.
 *
 * <pre>
 *   photo ──detector──▶ boxes around every car/truck
 *         ──crop each box (plus a margin)──▶ one small image per vehicle
 *         ──classifier, all crops in one batch──▶ one score per class, per vehicle
 *         ──softmax──▶ probabilities ──┬─ layered answer ▶ "Honda Civic, 8th gen (2006-2011)"
 *                                       └─ top 5 classes ▶ "Honda Civic (2009-2011)  45%", ...
 * </pre>
 *
 * <p>This mirrors training/predict.py, with one difference made for live mode: Python keeps
 * only the biggest vehicle, this returns ALL of them (biggest first), so a video frame with
 * three cars gets three labels.
 */
@Component
public class CarPipeline {

    private static final Logger log = LoggerFactory.getLogger(CarPipeline.class);

    /** How many guesses to return per vehicle. */
    private static final int TOP_K = 5;

    // Model input/output names, as exported by training/export_onnx.py (OnnxModels logs them at startup).
    private static final String DETECTOR_INPUT = "image";
    private static final String CLASSIFIER_INPUT = "input";
    private static final String CLASSIFIER_OUTPUT = "logits";

    private final OnnxModels models;
    private final ModelSettings settings;
    private final CarLabel[] labels;
    /** How sure a layer of the answer must be to be reported (see LayeredAnswer). */
    private final float answerThreshold;
    /** COCO category number -> our name for it, e.g. 3 -> "car", 8 -> "truck". */
    private final Map<Long, String> vehicleTypes = new HashMap<>();

    /**
     * Spring passes in the already-loaded models (OnnxModels is a @Component too). That's
     * "dependency injection": we ask for what we need in the constructor instead of creating
     * it ourselves, so there's exactly one copy of the ~190 MB of models in memory.
     *
     * <p>@Value pulls a setting out of application.properties (or the command line) and
     * converts it to the parameter's type: a Path, a float. Spring does that at startup.
     */
    public CarPipeline(OnnxModels models, @Value("${carid.models-dir}") Path modelsDir,
                       @Value("${carid.answer-threshold}") float answerThreshold) {
        this.models = models;
        this.answerThreshold = answerThreshold;
        Path dir = modelsDir.toAbsolutePath().normalize();

        // Jackson turns JSON into Java objects: the files' structure matches our records.
        JsonMapper json = JsonMapper.builder().build();
        settings = json.readValue(dir.resolve("preprocessing.json").toFile(), ModelSettings.class);
        labels = json.readValue(dir.resolve("labels.json").toFile(), CarLabel[].class);

        // Position in the array must equal the class number the model outputs.
        // If this ever fails, every prediction would be labeled with the wrong car.
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].index() != i) {
                throw new IllegalStateException("labels.json is out of order at position " + i);
            }
        }
        settings.detector().acceptLabels().forEach((name, id) -> vehicleTypes.put(id.longValue(), name));

        log.info("Loaded {} car classes; detector accepts {} at score >= {}; answer threshold {}",
                labels.length, settings.detector().acceptLabels().keySet(), settings.detector().minScore(),
                answerThreshold);
    }

    /**
     * Find and identify every vehicle in the photo.
     *
     * @param wholePhotoFallback what to do when no vehicle is found. true (single photos):
     *     classify the whole photo anyway, like predict.py. false (live video): return an
     *     empty list, because an empty street shouldn't get a made-up car name.
     */
    public List<CarResult> identify(RgbImage photo, boolean wholePhotoFallback) throws OrtException {
        long start = System.nanoTime();
        List<Detection> found = detect(photo);
        long detected = System.nanoTime();

        List<CarResult> results = new ArrayList<>();
        if (!found.isEmpty()) {
            List<RgbImage> crops = found.stream().map(d -> cropWithMargin(photo, d)).toList();
            List<Guesses> guesses = classify(crops);
            for (int i = 0; i < found.size(); i++) {
                Detection d = found.get(i);
                Guesses g = guesses.get(i);
                results.add(new CarResult(d.box(), d.type(), d.score(), true, g.answer(), g.top()));
            }
        } else if (wholePhotoFallback) {
            Box whole = new Box(0, 0, photo.width(), photo.height());
            Guesses g = classify(List.of(photo)).get(0);
            results.add(new CarResult(whole, null, 0f, false, g.answer(), g.top()));
        }
        long classified = System.nanoTime();

        // These timings are what decide whether live mode needs the GPU.
        log.info("Identified {}x{} photo: {} vehicle(s), detect {} ms, classify {} ms",
                photo.width(), photo.height(), found.size(),
                (detected - start) / 1_000_000, (classified - detected) / 1_000_000);
        return results;
    }

    // ------------------------------------------------------------ detector

    /** A detector hit before classification: where, what kind, how sure. */
    private record Detection(float left, float top, float right, float bottom, float score, String type) {
        double area() {
            return ((double) right - left) * ((double) bottom - top);
        }

        Box box() {
            return new Box(Math.round(left), Math.round(top), Math.round(right), Math.round(bottom));
        }
    }

    private List<Detection> detect(RgbImage photo) throws OrtException {
        float[] input = Preprocessing.detectorTensor(photo);
        long[] shape = {3, photo.height(), photo.width()};

        // OnnxTensor and Result hold native (C++) memory, so they're closed right after use.
        // try-with-resources closes them even if something throws, like a finally block.
        try (OnnxTensor tensor = OnnxTensor.createTensor(models.env(), FloatBuffer.wrap(input), shape);
             OrtSession.Result out = models.detector().run(Map.of(DETECTOR_INPUT, tensor))) {

            // Three parallel arrays, one entry per object the detector found (any kind:
            // people, dogs, cars...). getValue() copies the native data into Java arrays.
            float[][] boxes = (float[][]) out.get("boxes").orElseThrow().getValue();
            long[] cocoLabels = (long[]) out.get("labels").orElseThrow().getValue();
            float[] scores = (float[]) out.get("scores").orElseThrow().getValue();

            List<Detection> vehicles = new ArrayList<>();
            for (int i = 0; i < scores.length; i++) {
                String type = vehicleTypes.get(cocoLabels[i]);
                if (type != null && scores[i] >= settings.detector().minScore()) {
                    float[] b = boxes[i];
                    vehicles.add(new Detection(b[0], b[1], b[2], b[3], scores[i], type));
                }
            }
            // Biggest first, like predict.py: in a photo of "my car", it's usually the largest.
            vehicles.sort(Comparator.comparingDouble(Detection::area).reversed());
            return vehicles;
        }
    }

    /**
     * Crop to the box widened by crop_margin on each side, exactly like detect.crop_to_car():
     * the classifier learned from listing photos that show the whole car with some
     * surroundings, so a tight crop could cut off mirrors or bumpers.
     */
    private RgbImage cropWithMargin(RgbImage photo, Detection d) {
        double margin = settings.detector().cropMargin();
        double padX = ((double) d.right() - d.left()) * margin;
        double padY = ((double) d.bottom() - d.top()) * margin;
        // Same order as Python: widen, clamp to the photo, then truncate to whole pixels.
        int left = (int) Math.max(0, d.left() - padX);
        int top = (int) Math.max(0, d.top() - padY);
        int right = (int) Math.min(photo.width(), d.right() + padX);
        int bottom = (int) Math.min(photo.height(), d.bottom() + padY);
        // Guard against a degenerate (zero-width) box.
        right = Math.max(right, left + 1);
        bottom = Math.max(bottom, top + 1);
        return photo.crop(left, top, right, bottom);
    }

    // ------------------------------------------------------------ classifier

    /** What the classifier says about one image: the layered verdict and the top single classes. */
    private record Guesses(LayeredAnswer answer, List<Prediction> top) {
    }

    /** Classify several images in ONE model run (a "batch"), which is faster than one at a time. */
    private List<Guesses> classify(List<RgbImage> images) throws OrtException {
        ModelSettings.Classifier cfg = settings.classifier();
        int w = cfg.resize().width();
        int h = cfg.resize().height();
        int perImage = 3 * w * h;

        // Stack every image's tensor end to end: shape [count, 3, h, w].
        float[] batch = new float[images.size() * perImage];
        for (int i = 0; i < images.size(); i++) {
            RgbImage resized = Preprocessing.resizeBilinear(images.get(i), w, h);
            float[] one = Preprocessing.classifierTensor(resized, cfg.mean(), cfg.std());
            System.arraycopy(one, 0, batch, i * perImage, perImage);
        }
        long[] shape = {images.size(), 3, h, w};

        try (OnnxTensor tensor = OnnxTensor.createTensor(models.env(), FloatBuffer.wrap(batch), shape);
             OrtSession.Result out = models.classifier().run(Map.of(CLASSIFIER_INPUT, tensor))) {

            // One row of raw scores ("logits") per image, one score per class.
            float[][] logits = (float[][]) out.get(CLASSIFIER_OUTPUT).orElseThrow().getValue();
            List<Guesses> all = new ArrayList<>();
            for (float[] row : logits) {
                // The answer needs ALL the probabilities (it adds them up per layer),
                // so it's computed before topK throws away everything but the best 5.
                float[] probs = softmax(row);
                all.add(new Guesses(LayeredAnswer.decide(labels, probs, answerThreshold), topK(probs)));
            }
            return all;
        }
    }

    /**
     * Raw scores -> probabilities that add up to 1: exp(score) / sum of exp(all scores).
     * Subtracting the max first keeps exp() from overflowing; it doesn't change the result.
     */
    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - max);
            sum += exps[i];
        }
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) (exps[i] / sum);
        }
        return probs;
    }

    /** The TOP_K most likely classes, most likely first, with their names. */
    private List<Prediction> topK(float[] probs) {
        // Picking the max K times is simplest, and fast enough for ~1,000 values.
        boolean[] taken = new boolean[probs.length];
        List<Prediction> top = new ArrayList<>(TOP_K);
        for (int k = 0; k < Math.min(TOP_K, probs.length); k++) {
            int best = -1;
            for (int i = 0; i < probs.length; i++) {
                if (!taken[i] && (best < 0 || probs[i] > probs[best])) {
                    best = i;
                }
            }
            taken[best] = true;
            top.add(Prediction.of(labels[best], probs[best]));
        }
        return top;
    }
}
