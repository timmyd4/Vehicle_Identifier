package io.github.timmyd4.carid.server;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One class the classifier can predict, as listed in models/labels.json.
 *
 * <p>The classifier only outputs numbers: one score per position (836 in run 2). labels.json
 * (written by training/export_onnx.py) says which car each position means, in exactly the
 * order the model was trained with. Position 0 = "Acura CL (1997-1999)", and so on.
 *
 * <p>The display names were built in Python, so Java shows exactly the same text without
 * re-implementing Python's capitalization rules.
 *
 * <p>make / model / generation / phase are the LAYERS of the answer. Several classes share
 * a generation (pre + facelift), several generations share a model, and so on. Adding up
 * the probabilities of the classes that share a layer gives that layer's probability.
 *
 * @param makeName   the make as people write it: "Honda", "BMW", "Mercedes-Benz"
 * @param carName    make + model without years: "Honda Civic"
 * @param generation the table's name for this body, e.g. "8th gen", "E39", "NA" (Miata)
 * @param phase      "single" (never facelifted), "pre", "facelift", "facelift2" or "facelift3"
 */
// ignoreUnknown: if a future export adds fields, old Java code still loads the file.
@JsonIgnoreProperties(ignoreUnknown = true)
public record CarLabel(
        int index,
        // @JsonProperty maps Python's snake_case keys to Java's camelCase names.
        @JsonProperty("class_name") String className,
        @JsonProperty("display_name") String displayName,
        @JsonProperty("make_name") String makeName,
        @JsonProperty("car_name") String carName,
        String make,
        String model,
        @JsonProperty("year_start") int yearStart,
        @JsonProperty("year_end") int yearEnd,
        String generation,
        String phase) {

    /** True when this class is one look among several in its generation (pre, facelift, ...). */
    public boolean hasFacelifts() {
        return !"single".equals(phase);
    }
}
