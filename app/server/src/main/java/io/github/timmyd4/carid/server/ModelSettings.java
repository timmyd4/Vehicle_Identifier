package io.github.timmyd4.carid.server;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

/**
 * The parts of models/preprocessing.json the Java pipeline needs.
 *
 * <p>Why read these from the file instead of typing the numbers into Java: the Python export
 * writes this file next to the models, so the settings always match the models they came
 * with. If training changes the image size or the detector threshold, Java follows
 * automatically. One source of truth instead of two copies that can drift apart.
 *
 * <p>Jackson fills these nested records from the JSON's nested objects; the record and field
 * names mirror the JSON keys.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ModelSettings(Detector detector, Classifier classifier) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Detector(
            // {"car": 3, "truck": 8}: which COCO categories count as vehicles.
            @JsonProperty("accept_labels") Map<String, Integer> acceptLabels,
            // Ignore detections the detector is less sure of than this (0.5 = 50%).
            @JsonProperty("min_score") double minScore,
            // Widen each box by this fraction of its size on every side before cropping.
            @JsonProperty("crop_margin") double cropMargin) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Classifier(Resize resize, float[] mean, float[] std) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Resize(int width, int height) {
    }
}
