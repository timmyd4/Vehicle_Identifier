package io.github.timmyd4.carid.server;

import java.util.List;

/**
 * One vehicle found in a photo or video frame, with the classifier's best guesses for it.
 *
 * <p>These records are also the JSON the server will send back: Spring turns each record
 * field into a JSON key automatically (record -> {"box": ..., "vehicleType": ..., ...}).
 *
 * @param box          where the vehicle is, in the photo's pixels (for drawing an outline)
 * @param vehicleType  what the detector called it: "car" or "truck"
 * @param detectorScore how sure the detector is that a vehicle is there (0-1)
 * @param detected     false when no vehicle was found and the whole photo was classified instead
 * @param answer       the verdict: the most detailed layer the model is sure of (see LayeredAnswer)
 * @param predictions  the top single classes, most likely first: the raw detail behind the answer
 */
public record CarResult(Box box, String vehicleType, float detectorScore, boolean detected,
                        LayeredAnswer answer, List<Prediction> predictions) {

    /** A rectangle in pixels: left/top inclusive, right/bottom exclusive. */
    public record Box(int left, int top, int right, int bottom) {
    }

    /**
     * One guess: which car, and how likely (0-1). The probabilities of all classes
     * add up to 1, so 0.82 means "82% of the model's belief is on this one".
     */
    public record Prediction(String name, String make, String model, int yearStart, int yearEnd,
                             float probability) {

        static Prediction of(CarLabel label, float probability) {
            return new Prediction(label.displayName(), label.make(), label.model(),
                    label.yearStart(), label.yearEnd(), probability);
        }
    }
}
