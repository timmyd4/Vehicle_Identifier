package io.github.timmyd4.carid.server;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One class the classifier can predict, as listed in models/labels.json.
 *
 * <p>The classifier only outputs numbers: 1,168 scores, one per position. labels.json
 * (written by training/export_onnx.py) says which car each position means, in exactly the
 * order the model was trained with. Position 0 = "Acura CL (1998-2000)", and so on.
 *
 * <p>The display names were built in Python, so Java shows exactly the same text without
 * re-implementing Python's capitalization rules.
 */
// ignoreUnknown: if a future export adds fields, old Java code still loads the file.
@JsonIgnoreProperties(ignoreUnknown = true)
public record CarLabel(
        int index,
        // @JsonProperty maps Python's snake_case keys to Java's camelCase names.
        @JsonProperty("class_name") String className,
        @JsonProperty("display_name") String displayName,
        String make,
        String model,
        @JsonProperty("year_start") int yearStart,
        @JsonProperty("year_end") int yearEnd) {
}
